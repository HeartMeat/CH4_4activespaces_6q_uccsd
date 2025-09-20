import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from scipy.linalg import eigh
from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SciPyOptimizer

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.units import DistanceUnit

from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt

# =============================================================================
# [전역 파라미터] — 최적화/알고리즘/시각화
#   - 아래 값들은 재현성 및 수렴 품질을 제어. 실험 시 필요한 것만 조정.
# =============================================================================
VQE_MAXITER  = 10000          # VQE(L-BFGS-B) 최대 반복 수
VQE_FTOL     = 1e-9           # VQE 수렴 허용오차
VQE_RESTARTS = 2              # 서로 다른 초기점으로 재시도 횟수(최적 에너지 선택)

VQD_MAXITER   = 10000         # VQD(L-BFGS-B) 각 단계 최대 반복
VQD_FTOL      = 1e-9          # VQD 수렴 허용오차
VQD_RESTARTS  = 2             # VQD도 다중 초기화로 로컬 미니마 회피
VQD_BETAS     = (5.0, 5.0)    # VQD 직교(겹침) 페널티(β): 1단계(T1), 2단계(S1)

# qEOM 설정:
#   - QEOM_CUT_NORM : 생성/소멸 조합으로 만든 보조벡터(φ_k, φ†_k)의 노름 컷오프
#   - QEOM_S_EIGCUT : 메트릭 행렬 S의 작은 고윳값 제거(수치적 불안정 억제)
#   - QEOM_OMEGA_MIN: 양의 천이에너지(ω) 최소 컷(수치잡음으로 생기는 ~0 모드 제거)
QEOM_CUT_NORM   = 1e-6
QEOM_S_EIGCUT   = 1e-6
QEOM_OMEGA_MIN  = 1e-3

# 시각화 옵션
SHOW_PLOTS = True             # 플롯 화면 표시 여부
SAVE_FIGS  = False            # 플롯 파일 저장 여부
FIG_DPI    = 150

# 기타 출력 제어
PRINT_FCI_FIRST_FEW = True    # FCI 섹터 내 낮은 에너지 몇 개 출력

# 난수/단위
RNG_SEED = 42
np.random.seed(RNG_SEED)
algorithm_globals.random_seed = RNG_SEED

HARTREE_TO_EV = 27.211386245981
def ha_to_ev(x: float) -> float:
    """Hartree → eV 변환."""
    return x * HARTREE_TO_EV


@dataclass
class Spectrum:
    """단일 분자에 대한 에너지 준위 묶음.
    - E0: 바닥상태 S0 (Hartree)
    - E1: 첫 삼중항 T1 (Hartree)
    - E2: 첫 단일항 S1 (Hartree)
    """
    E0: float
    E1: float
    E2: float


# =============================================================================
# 기하/문제 구성
# =============================================================================
def ch4_geometry() -> List[Tuple[str, Tuple[float, float, float]]]:
    """CH4의 이상적 정사면체(tetrahedral) 좌표(Å).
    - C–H 결합길이 r≈1.09Å, 중심 C (0,0,0), H는 (±a, ±a, ±a) 조합 중 짝수 부호.
    """
    r = 1.09
    a = r / np.sqrt(3.0)
    Hs = [( a,  a,  a), ( a, -a, -a), (-a,  a, -a), (-a, -a,  a)]
    return [("C", (0.0, 0.0, 0.0))] + [("H", xyz) for xyz in Hs]


def build_problem_ch4_sto3g() -> Tuple[ElectronicStructureProblem, List[int]]:
    """CH4(STO-3G) 전자구조 문제 구성 + 4개 공간궤도(active space) 선택.
    - 논문은 HOMO/LUMO 2o/2e를 사용했지만, 여기서는 HOMO-1..LUMO+1 (4o/4e)로 확장.
    - 알고리즘 검증용 토이 문제
    - 선택 절차:
        1) PySCFDriver로 전체 문제 구성
        2) α전자 수로부터 HOMO 인덱스(homo=nα-1) 계산
        3) [homo-1, homo, lumo, lumo+1]을 active로 선택(경계 검사)
        4) ActiveSpaceTransformer로 4e/4o 문제로 축소
    """
    geom = "; ".join([f"{s} {c[0]} {c[1]} {c[2]}" for s, c in ch4_geometry()])
    driver = PySCFDriver(atom=geom, basis="sto3g", unit=DistanceUnit.ANGSTROM)
    base_problem = driver.run()

    n_alpha, _ = base_problem.num_particles
    ns = base_problem.num_spatial_orbitals
    homo = n_alpha - 1
    active_orbs = [homo - 1, homo, homo + 1, homo + 2]
    active_orbs = [i for i in active_orbs if 0 <= i < ns]

    if len(active_orbs) != 4:
        raise ValueError(
            f"[ERROR] Active space selection failed: expected 4 spatial orbitals, got {len(active_orbs)} "
            f"within ns={ns}. Selected={active_orbs}"
        )

    # 4전자/4공간궤도로 제한(명시적 인덱스 지정으로 HOMO-1..LUMO+1 고정)
    trans = ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=4, active_orbitals=active_orbs)
    problem = trans.transform(base_problem)

    print(f"[LOG] Active space indices (spatial): {active_orbs}")
    print(f"[LOG] Particles (α,β): {problem.num_particles}, spatial-orbitals={problem.num_spatial_orbitals}")
    return problem, active_orbs


# =============================================================================
# 맵핑/연산자 유틸
# =============================================================================
def _map_qubit_matrix(mapper, ferm_op: FermionicOp, register_length: int, dim: int) -> np.ndarray:
    """FermionicOp → qubit 연산자 → 밀집행렬.
    - 매핑 실패/빈 연산자는 0행렬 반환해 수치 안전성 확보.
    """
    qop = mapper.map(ferm_op, register_length=register_length)
    if qop is None:
        return np.zeros((dim, dim), dtype=complex)
    try:
        if hasattr(qop, "size") and qop.size == 0:
            return np.zeros((dim, dim), dtype=complex)
        return qop.to_matrix()
    except Exception:
        return np.zeros((dim, dim), dtype=complex)


def build_tapered_mapper(problem: ElectronicStructureProblem):
    """Parity 매핑 + Z2 테이퍼링(tapering) 구성.
    - qiskit-nature의 problem.get_tapered_mapper 사용(대칭 섹터 자동 선택).
    """
    base_mapper = ParityMapper()
    if not hasattr(problem, "get_tapered_mapper"):
        raise RuntimeError("This qiskit-nature version lacks problem.get_tapered_mapper; cannot perform Z₂ tapering.")
    tapered_mapper = problem.get_tapered_mapper(base_mapper)
    print("[LOG] Using Parity + Z₂ symmetry tapering (via problem.get_tapered_mapper).")
    return base_mapper, tapered_mapper


def qubit_hamiltonians(problem: ElectronicStructureProblem,
                       base_mapper: ParityMapper,
                       tapered_mapper) -> Tuple[SparsePauliOp, int, int, int]:
    """해밀토니안(2차양자 → 큐비트)을 기본/테이퍼 버전으로 생성하고 요약 정보 출력."""
    op2 = problem.hamiltonian.second_q_op()
    Hq_base = base_mapper.map(op2)
    Hq_tap  = tapered_mapper.map(op2)
    n_terms_tap = int(np.size(Hq_tap.coeffs))
    n_full  = Hq_base.num_qubits
    n_tap   = Hq_tap.num_qubits
    print(f"[LOG] Qubit Hamiltonian (base)   — qubits: {n_full}, terms={int(np.size(Hq_base.coeffs))}")
    print(f"[LOG] Qubit Hamiltonian (taper)  — qubits: {n_tap},  terms={n_terms_tap}")
    if n_full - n_tap != 2 or n_tap != 6:
        print(f"[WARN] Expected 2-qubit Z₂ tapering to 6 qubits, got base={n_full}, tapered={n_tap}. Proceeding.")
    return Hq_tap, n_terms_tap, n_full, n_tap


def _state_from_params(ansatz: QuantumCircuit, theta: np.ndarray) -> Statevector:
    """파라미터 θ를 ansatz에 바인딩하여 상태벡터(Statevector) 생성."""
    bind = {p: v for p, v in zip(ansatz.parameters, theta)}
    circ = ansatz.assign_parameters(bind, inplace=False)
    return Statevector.from_instruction(circ)


def _expect(A: np.ndarray, psi: np.ndarray) -> float:
    """기대값 ⟨ψ|A|ψ⟩의 실수부."""
    return float(np.real(np.vdot(psi, A @ psi)))


def _number_ops_matrices(problem: ElectronicStructureProblem, mapper, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """α/β 전자수 연산자 Nα, Nβ의 큐비트 행렬 생성(섹터 필터링에 사용)."""
    ns = problem.num_spatial_orbitals
    reg_len = 2 * ns
    terms_na, terms_nb = {}, {}
    for p in range(ns):
        terms_na[f"+_{p} -_{p}"] = 1.0
        terms_nb[f"+_{p+ns} -_{p+ns}"] = 1.0
    Nalpha = _map_qubit_matrix(mapper, FermionicOp(terms_na, num_spin_orbitals=reg_len), reg_len, dim)
    Nbeta  = _map_qubit_matrix(mapper, FermionicOp(terms_nb, num_spin_orbitals=reg_len), reg_len, dim)
    return Nalpha, Nbeta


# =============================================================================
# 정확해(FCI) 준위(섹터 필터링)
# =============================================================================
def exact_fci_sector(H: np.ndarray,
                     Nalpha: np.ndarray, Nbeta: np.ndarray,
                     nalpha: int, nbeta: int,
                     show_first: int = 6) -> Spectrum:
    """전체(테이퍼된) 해밀토니안의 정확한 대각화 후, (Nα, Nβ) 섹터에 속하는
    가장 낮은 3개 준위를 S0/T1/S1로 선택.
    """
    w, V = eigh(H)
    states = [V[:, k] for k in range(V.shape[1])]
    na_vals = np.array([float(np.real(np.vdot(v, Nalpha @ v))) for v in states])
    nb_vals = np.array([float(np.real(np.vdot(v, Nbeta  @ v))) for v in states])
    tolN = 1e-8
    mask = (np.abs(na_vals - nalpha) < tolN) & (np.abs(nb_vals - nbeta) < tolN)

    idxs = [i for i in np.argsort(w) if mask[i]]
    if len(idxs) < 3:
        raise RuntimeError("Not enough eigenstates in the (Nα,Nβ) sector to pick E0,E1,E2.")
    E0 = float(w[idxs[0]])
    E1 = float(w[idxs[1]])
    E2 = float(w[idxs[2]])

    print("[LOG] FCI(full-H, tapered): done (sector-filtered by ⟨Nα⟩,⟨Nβ⟩)")
    if PRINT_FCI_FIRST_FEW:
        print(f"[LOG] FCI spectrum (first few in sector Nα={nalpha}, Nβ={nbeta}):")
        shown = 0
        for i in idxs[:show_first]:
            print(f"   {shown:02d}: E={w[i]: .8f} Ha  (⟨Nα⟩≈{na_vals[i]:.3f}, ⟨Nβ⟩≈{nb_vals[i]:.3f})")
            shown += 1
    print(f"[LOG] FCI picks: T1={E1:.8f} Ha, S1={E2:.8f} Ha")
    return Spectrum(E0=E0, E1=E1, E2=E2)


# =============================================================================
# VQE: UCCSD(HF 참조)로 S0 최적화
# =============================================================================
def build_uccsd(problem: ElectronicStructureProblem, tapered_mapper) -> QuantumCircuit:
    """UCCSD(ansatz) + Hartree–Fock 초기상태(HF)를 테이퍼 맵퍼 기준으로 생성."""
    num_spatial_orbitals = problem.num_spatial_orbitals
    num_particles = problem.num_particles
    hf = HartreeFock(num_spatial_orbitals, num_particles, tapered_mapper)
    uccsd = UCCSD(
        qubit_mapper=tapered_mapper,
        num_particles=num_particles,
        num_spatial_orbitals=num_spatial_orbitals,
        initial_state=hf
    )
    return uccsd


def _extract_theta_from_vqe(result, ansatz: QuantumCircuit) -> np.ndarray:
    """VQE 결과 객체로부터 최적 파라미터 벡터 θ를 추출."""
    if hasattr(result, "optimal_parameters") and isinstance(result.optimal_parameters, dict):
        return np.array([result.optimal_parameters[p] for p in ansatz.parameters], dtype=float)
    if hasattr(result, "optimal_point"):
        return np.array(result.optimal_point, dtype=float)
    raise RuntimeError("VQE optimal parameters not found.")


def _run_vqe_once(Hq: SparsePauliOp, ansatz: QuantumCircuit, x0: np.ndarray) -> Tuple[float, np.ndarray]:
    """주어진 초기점 x0에서 VQE 한 번 실행."""
    est = StatevectorEstimator()
    opt = SciPyOptimizer(method="L-BFGS-B", options={"maxiter": VQE_MAXITER, "ftol": VQE_FTOL})
    vqe = VQE(estimator=est, ansatz=ansatz, optimizer=opt, initial_point=x0)
    res = vqe.compute_minimum_eigenvalue(Hq)
    e0 = float(np.real(res.eigenvalue))
    theta = _extract_theta_from_vqe(res, ansatz)
    print(f"      [seed] E0={e0:.8f} Ha")
    return e0, theta


def vqe_ground(Hq: SparsePauliOp, ansatz: QuantumCircuit,
               restarts: int = VQE_RESTARTS) -> tuple[float, np.ndarray, np.ndarray]:
    """VQE로 바닥상태(S0) 에너지/파라미터/상태를 추정.
    - 서로 다른 초기점(restarts) 중 최적값 선택(로컬 최소 회피).
    """
    n = len(ansatz.parameters)
    print(f"[LOG] VQE(L-BFGS-B) with restarts={restarts}, ansatz='{ansatz.name}' "
          f"(params={n}, qubits={ansatz.num_qubits})")
    best = (np.inf, None)
    seeds = [np.zeros(n)]
    rng = np.random.default_rng(RNG_SEED)
    for _ in range(max(0, restarts-1)):
        seeds.append(0.1 * rng.standard_normal(n))
    t0 = time.time()
    for x0 in seeds:
        e, th = _run_vqe_once(Hq, ansatz, x0)
        if e < best[0]:
            best = (e, th)
    dt = time.time() - t0
    e0 = best[0]
    theta = best[1]
    psi0 = _state_from_params(ansatz, theta).data
    print(f"[LOG] VQE(best): E0={e0:.8f} Ha, time={dt:.2f}s, ftol={VQE_FTOL}")
    return e0, theta, psi0


# =============================================================================
# qEOM(full) — VQE 참조상태 기준의 방정식-운동(EOM) 고유값 문제
# =============================================================================
def _pool_spin_adapted(problem: ElectronicStructureProblem) -> Tuple[List[FermionicOp], Dict[str, int]]:
    """스핀 적응(spin-adapted)된 연산자 풀(싱글/더블 유사)을 구성.
    - S(Ms)=0 대칭을 반영한 조합으로 qEOM 서브공간 크기를 줄이고 안정화.
    """
    ns = problem.num_spatial_orbitals
    na, _ = problem.num_particles
    occ = list(range(min(na, ns)))
    virt = list(range(len(occ), ns))

    ops: List[FermionicOp] = []
    count = {"singles_like":0, "pair_doubles":0, "inter_doubles":0}

    def fop(term_dict):
        return FermionicOp(term_dict, num_spin_orbitals=2*ns)

    inv_sqrt2 = 1/np.sqrt(2)

    # (1) single 유사(α, β 대칭/반대칭 조합)
    for i in occ:
        for r in virt:
            op1 = inv_sqrt2 * fop({f"+_{r} -_{i}": 1.0}) + inv_sqrt2 * fop({f"+_{r+ns} -_{i+ns}": 1.0})
            op2 = inv_sqrt2 * fop({f"+_{r} -_{i}": 1.0}) - inv_sqrt2 * fop({f"+_{r+ns} -_{i+ns}": 1.0})
            ops.append(op1); ops.append(op2); count["singles_like"] += 2

    # (2) 같은 공간궤도 내 αβ 쌍-여기(pair doubles)
    for i in occ:
        for r in virt:
            ops.append(fop({f"+_{r} +_{r+ns} -_{i+ns} -_{i}": 1.0}))
            count["pair_doubles"] += 1

    # (3) 서로 다른 공간궤도 간 더블(inter-orbital doubles): 대칭/반대칭
    for a in range(len(occ)):
        for b in range(a+1, len(occ)):
            i, j = occ[a], occ[b]
            for c in range(len(virt)):
                for d in range(c+1, len(virt)):
                    r, s = virt[c], virt[d]
                    t1 = {f"+_{r} +_{s+ns} -_{j+ns} -_{i}": 1.0}
                    t2 = {f"+_{s} +_{r+ns} -_{i+ns} -_{j}": 1.0}
                    ops.append(inv_sqrt2 * fop(t1) + inv_sqrt2 * fop(t2))
                    ops.append(inv_sqrt2 * fop(t1) - inv_sqrt2 * fop(t2))
                    count["inter_doubles"] += 2

    total_ops = len(ops)
    print(f"[LOG] Operator pool (Ms=0-like): total={total_ops} | singles≈{count['singles_like']}, "
          f"pairD={count['pair_doubles']}, interD≈{count['inter_doubles']}")
    return ops, count


def qeom_excited_full(H: np.ndarray,
                      E0: float,
                      psi0: np.ndarray,
                      problem: ElectronicStructureProblem,
                      mapper) -> Tuple[float, float, Dict]:
    """qEOM(full)로 T1/S1 추정.
    절차:
      1) 스핀적응 풀 {O_k}로부터 |φ_k⟩=O_k|ψ0⟩, |φ̄_k⟩=O_k†|ψ0⟩ 생성
      2) 노름 작은 벡터 제거(QEOM_CUT_NORM)
      3) 블록 행렬 S, A 구성 후 EOM 일반화 고유값 문제를 정규화(whitening)
      4) 양의 천이에너지(ω) 중 작은 2개(≥ QEOM_OMEGA_MIN)를 T1/S1로 선택
    반환: (E_T1, E_S1, 수치진단 dict)
    """
    ns = problem.num_spatial_orbitals
    reg_len = 2 * ns

    ops, _ = _pool_spin_adapted(problem)
    Omats = [_map_qubit_matrix(mapper, op, reg_len, H.shape[0]) for op in ops]
    Odags = [m.conj().T for m in Omats]

    PHI   = [m  @ psi0 for m  in Omats]
    PHI_d = [md @ psi0 for md in Odags]

    norms = np.array([np.sqrt(np.linalg.norm(PHI[k])**2 + np.linalg.norm(PHI_d[k])**2) for k in range(len(ops))])
    keep  = norms > QEOM_CUT_NORM
    if not np.any(keep):
        print("[LOG] qEOM(full): empty subspace after norm cut.")
        return np.nan, np.nan, {"kept_ops":0, "total_ops":len(ops)}

    PHI  = [PHI[i]   for i in range(len(ops)) if keep[i]]
    PHId = [PHI_d[i] for i in range(len(ops)) if keep[i]]
    K = len(PHI)

    P = np.stack(PHI, axis=1)
    Q = np.stack(PHId, axis=1)
    B = np.concatenate([P, Q], axis=1)

    # 메트릭 S, 해밀토니안 A 블록 구성(EOM 정식)
    S_PP = P.conj().T @ P
    S_PM = P.conj().T @ Q
    S_MP = S_PM.conj().T
    S_MM = Q.conj().T @ Q
    S = np.block([[S_PP, S_PM],
                  [S_MP, S_MM]])
    S = 0.5 * (S + S.conj().T)  # 수치적 대칭화

    HP = H @ P
    HQ = H @ Q
    A_PP = P.conj().T @ HP
    A_PM = P.conj().T @ HQ
    A_MP = Q.conj().T @ HP
    A_MM = Q.conj().T @ HQ
    A = np.block([[A_PP, A_PM],
                  [A_MP, A_MM]])
    A = 0.5 * (A + A.conj().T)
    A -= E0 * S                    # E=E0 기준으로 이동(A ← A − E0 S)

    # S의 작은 고윳값 제거로 정규화(whitening)
    evals, U = eigh(S)
    keep_s = evals > QEOM_S_EIGCUT
    n_drop = int((~keep_s).sum())
    if np.count_nonzero(keep_s) == 0:
        print(f"[LOG] qEOM(full): S singular; no eigenvalues > {QEOM_S_EIGCUT}.")
        return np.nan, np.nan, {"kept_ops":K, "total_ops":len(ops), "S_rank":0}

    U_k = U[:, keep_s]
    lam_k = evals[keep_s]
    Sinvhalf = U_k @ np.diag(1.0/np.sqrt(lam_k)) @ U_k.conj().T

    # 진단 정보(컨디션, 노름 등)
    try:
        condS = float(np.linalg.cond(S))
    except Exception:
        condS = float("inf")
    B_norm2 = float(np.linalg.norm(B, 2))
    lam_min = float(evals.min()) if evals.size else float("nan")
    lam_max = float(evals.max()) if evals.size else float("nan")

    # 정규화된 고유값 문제 M y = ω y
    M = Sinvhalf.conj().T @ A @ Sinvhalf
    M = 0.5 * (M + M.conj().T)

    w, _ = eigh(M)
    order = np.argsort(w)
    w = w[order]

    # 양의 ω에서 두 개(≥ ω_min) 선택 → T1, S1
    omegas = [float(om) for om in w if om >= QEOM_OMEGA_MIN]
    if len(omegas) < 2:
        print("[WARN] qEOM(full): fewer than 2 positive ω above cutoff; results may be incomplete.")
        while len(omegas) < 2 and len(w) > len(omegas):
            candidate = float(w[len(omegas)])
            omegas.append(candidate)

    E1 = float(E0 + omegas[0])  # T1
    E2 = float(E0 + omegas[1])  # S1

    print(f"[LOG] qEOM(full): kept_ops={K}/{len(ops)}, block_dim={2*K}, "
          f"S_eig_kept={len(lam_k)}, S_eig_dropped={n_drop}, "
          f"λ_min/max(S)={lam_min:.3e}/{lam_max:.3e}, cond(S)≈{condS:.3e}, ||B||₂≈{B_norm2:.3e}, "
          f"ω_min={QEOM_OMEGA_MIN}")
    return E1, E2, {
        "kept_ops": K,
        "total_ops": len(ops),
        "block_dim": 2*K,
        "S_rank": len(lam_k),
        "omegas": w.tolist(),
        "S_eigs_kept_min": float(lam_k.min()) if len(lam_k) else float("nan"),
        "omega_min_cut": QEOM_OMEGA_MIN,
        "condS": condS,
        "lam_min": lam_min,
        "lam_max": lam_max,
        "B_norm2": B_norm2,
    }


# =============================================================================
# VQD — 직교 페널티 기반 여기상태 최적화
# =============================================================================
def vqd_excited(H: np.ndarray,
                problem: ElectronicStructureProblem, mapper,
                ansatz: QuantumCircuit, theta0: np.ndarray,
                betas=VQD_BETAS,
                restarts=VQD_RESTARTS, maxiter=VQD_MAXITER) -> tuple[float, float, float]:
    """VQD로 (S0, T1, S1) 순차 최적화.
    - 단계 k(=1,2)에서 비용함수:  E(θ) + β_k ∑ |⟨ψ_ref|ψ(θ)⟩|^2
      (이미 찾은 상태들과의 겹침(Overlap)을 벌점으로 추가해 직교 유도)
    - 여기서는 상태 심벌만 간단히 표기하고 분리된 측정 항 구성은 생략(순수 행렬식 사용).
    """
    rng = np.random.default_rng(RNG_SEED)
    psi0 = _state_from_params(ansatz, theta0)
    E0 = _expect(H, psi0.data)

    found_states = [psi0]
    energies = [E0]

    for step, beta in enumerate(betas, start=1):
        n = len(ansatz.parameters)

        def objective(x):
            psi = _state_from_params(ansatz, x)
            val = _expect(H, psi.data)
            # 직교 페널티(이미 찾은 모든 상태에 대해 겹침 제곱 가산)
            for b, ref in zip([beta]*len(found_states), found_states):
                ov = complex(np.vdot(ref.data, psi.data))
                val += float(b * (ov.conjugate()*ov).real)
            return val

        best = (np.inf, None)
        x0_base = np.zeros(n)
        for _ in range(restarts):
            x0 = x0_base + 0.1 * rng.standard_normal(n)
            res = minimize(objective, x0, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": VQD_FTOL})
            if res.fun < best[0]:
                best = (res.fun, res.x)

        theta_opt = best[1]
        psi_k = _state_from_params(ansatz, theta_opt)
        Ek = _expect(H, psi_k.data)
        found_states.append(psi_k)
        energies.append(Ek)

        print(f"[LOG] VQD step {step} [Excited #{step}] (L-BFGS-B): "
              f"E={Ek:.8f} Ha, obj*={best[0]:.8f}, maxiter={maxiter}, ftol={VQD_FTOL}, "
              f"restarts={restarts}, beta={beta}")

    return float(energies[0]), float(energies[1]), float(energies[2])


# =============================================================================
# 리포팅/플롯
# =============================================================================
def report_line(tag, E0, E1, E2, ref: Optional[Spectrum] = None, extra: str = ""):
    """한 줄 요약 문자열 구성(에너지 차이는 eV로)."""
    t1 = ha_to_ev(E1 - E0)
    s1 = ha_to_ev(E2 - E0)
    dEST  = ha_to_ev(E2 - E1)
    s = (f"{tag:16s} | S0 {E0: .8f} Ha | T1–S0 {t1:6.3f} eV | S1–S0 {s1:6.3f} eV | "
         f"ΔE_ST {dEST:5.3f} eV")
    if ref is not None:
        s += (f"\n{'':16s}   dev vs FCI: "
              f"Δ(T1–S0) {ha_to_ev((E1 - E0) - (ref.E1 - ref.E0)):+6.3f} eV, "
              f"Δ(S1–S0) {ha_to_ev((E2 - E0) - (ref.E2 - ref.E0)):+6.3f} eV, "
              f"Δ(ΔE_ST) {ha_to_ev((E2 - E1) - (ref.E2 - ref.E1)):+6.3f} eV")
    if extra:
        s += f"\n{'':16s}   {extra}"
    return s


def plot_ansatz_circuit(ansatz: QuantumCircuit, title: str):
    """Ansatz 회로를 Matplotlib로 렌더링(옵션 저장/표시)."""
    fig = ansatz.draw(output="mpl", fold=-1)
    fig.suptitle(title, y=0.98)
    if SAVE_FIGS:
        fig.savefig("fig0_ansatz.png", dpi=FIG_DPI)
    if not SHOW_PLOTS:
        plt.close(fig)


def plot_all(all_results: Dict[str, Dict[str, Spectrum]], title_suffix: str = ""):
    """T1/S1 여기 에너지 막대, ΔE_ST 상관 산점, 에너지 레벨 비교 플롯."""
    mols = list(all_results.keys())
    mol = mols[0]
    fci = all_results[mol]["FCI"]
    qe  = all_results[mol]["qEOM"]
    vq  = all_results[mol]["VQD"]

    labels = ["T1", "S1"]
    x = np.arange(2); w = 0.25

    # (1) 여기 에너지(eV)
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
    ax.bar(x - w, [ha_to_ev(fci.E1 - fci.E0), ha_to_ev(fci.E2 - fci.E0)], w, label="FCI")
    ax.bar(x,      [ha_to_ev(qe.E1  - qe.E0),  ha_to_ev(qe.E2  - qe.E0)],  w, label="qEOM(full)")
    ax.bar(x + w,  [ha_to_ev(vq.E1  - vq.E0),  ha_to_ev(vq.E2  - vq.E0)],  w, label="VQD")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Excitation energy [eV]")
    ax.set_title(f"{mol} — Excitation energies (T1, S1){title_suffix}")
    ax.grid(True, alpha=0.3); ax.legend()
    if SAVE_FIGS: fig.savefig("fig1_bar_excitation_eV.png", dpi=FIG_DPI)
    if not SHOW_PLOTS: plt.close(fig)

    # (2) ΔE_ST 상관(FCI vs 알고리즘)
    fig2, ax2 = plt.subplots(1, 1, figsize=(5.7, 4.6), constrained_layout=True)
    fci_gap = ha_to_ev(fci.E2 - fci.E1)
    qe_gap  = ha_to_ev(qe.E2  - qe.E1)
    vq_gap  = ha_to_ev(vq.E2  - vq.E1)
    mn = min(fci_gap, qe_gap, vq_gap) - 0.2
    mx = max(fci_gap, qe_gap, vq_gap) + 0.2
    ax2.scatter([fci_gap], [qe_gap], marker="o", label="qEOM(full)")
    ax2.scatter([fci_gap], [vq_gap], marker="s", label="VQD")
    ax2.plot([mn, mx], [mn, mx], "k--", lw=1)
    ax2.set_xlabel("FCI ΔE_ST [eV]")
    ax2.set_ylabel("Algorithm ΔE_ST [eV]")
    ax2.set_title(f"{mol} — ΔE_ST correlation{title_suffix}")
    ax2.grid(True, alpha=0.3); ax2.legend(loc="best")
    if SAVE_FIGS: fig2.savefig("fig2_delta_correlation.png", dpi=FIG_DPI)
    if not SHOW_PLOTS: plt.close(fig2)

    # (3) 에너지 레벨(FCI 기준 상대)
    fig3, ax3 = plt.subplots(1, 1, figsize=(6.7, 4.2), constrained_layout=True)
    def draw(x0, levels, label):
        for E in levels:
            ax3.hlines(ha_to_ev(E - fci.E0), x0 - 0.3, x0 + 0.3, lw=2)
        ax3.text(x0, ha_to_ev(levels[-1] - fci.E0) + 0.05, label, ha="center")
    draw(0.8, [fci.E0, fci.E1, fci.E2], "FCI")
    draw(1.6, [qe.E0,  qe.E1,  qe.E2 ], "qEOM(full)")
    draw(2.4, [vq.E0,  vq.E1,  vq.E2 ], "VQD")
    ax3.set_title(f"{mol} — Levels (relative to FCI S0){title_suffix}")
    ax3.set_ylabel("Energy [eV]")
    ax3.set_xticks([0.8, 1.6, 2.4]); ax3.set_xticklabels(["FCI", "qEOM", "VQD"])
    ax3.grid(True, axis="y", alpha=0.3)
    if SAVE_FIGS:
        fig3.savefig("fig3_levels_relative.png", dpi=FIG_DPI)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig3)


# ===== 표 출력 유틸 =====
def _table_border(widths):
    return "+" + "+".join("-"*(w+2) for w in widths) + "+"


def _table_row(cells, widths, align_right=True):
    parts = []
    for v, w in zip(cells, widths):
        s = f"{v}"
        parts.append(s.rjust(w) if align_right else s.ljust(w))
    return "| " + " | ".join(parts) + " |"


def print_energy_table(spec_fci: Spectrum, spec_qe: Spectrum, spec_vq: Spectrum, meta: str = ""):
    """최종 에너지 표를 콘솔에 ASCII 테이블로 출력."""
    headers = ["Method", "S0 (Ha)", "T1 (Ha)", "S1 (Ha)", "ΔE_ST (mHa)"]
    widths  = [10, 14, 14, 14, 14]

    def r(method, sp: Spectrum):
        d_mha = (sp.E2 - sp.E1) * 1000.0  # mHa
        return [
            method,
            f"{sp.E0: .8f}",
            f"{sp.E1: .8f}",
            f"{sp.E2: .8f}",
            f"{d_mha:7.3f}",
        ]

    rows = [r("FCI", spec_fci), r("qEOM", spec_qe), r("VQD", spec_vq)]

    print("\n[RESULT TABLE | Energies (Ha, mHa)]")
    print(_table_border(widths))
    print(_table_row(headers, widths, align_right=False))
    print(_table_border(widths))
    for row in rows:
        print(_table_row(row, widths, align_right=True))
    print(_table_border(widths))
    if meta:
        print(f"  {meta}")


# =============================================================================
# 메인
# =============================================================================
def main():
    print(f"[LOG] Reproducibility seed = {RNG_SEED}")
    problem, _active_orbs = build_problem_ch4_sto3g()

    base_mapper, tapered_mapper = build_tapered_mapper(problem)

    Hq, n_terms, n_full, n_tap = qubit_hamiltonians(problem, base_mapper, tapered_mapper)
    print(f"[LOG] Qubits: base={n_full} → tapered={n_tap}")

    H = Hq.to_matrix()
    dim = H.shape[0]
    Nalpha, Nbeta = _number_ops_matrices(problem, tapered_mapper, dim)
    na, nb = problem.num_particles

    # 정확해(FCI): (Nα, Nβ) 섹터 필터링 후 S0/T1/S1 선택
    spec_fci = exact_fci_sector(H, Nalpha, Nbeta, nalpha=na, nbeta=nb)

    # VQE(S0): UCCSD(HF 초기상태 포함)
    ansatz = build_uccsd(problem, tapered_mapper)
    if SHOW_PLOTS or SAVE_FIGS:
        plot_ansatz_circuit(
            ansatz,
            f"Ansatz: UCCSD(HF ref, excitations='sd')  |  tapered qubits = {Hq.num_qubits}"
        )
    e0_vqe, theta, psi0 = vqe_ground(Hq, ansatz, restarts=VQE_RESTARTS)

    # qEOM(full): ω ≥ ω_min에서 작은 2개 → (T1, S1)
    e1_qeom, e2_qeom, qeom_info = qeom_excited_full(H, e0_vqe, psi0, problem, tapered_mapper)

    # VQD: 직교 페널티로 (T1, S1) 탐색
    e0_vqd, e1_vqd, e2_vqd = vqd_excited(
        H, problem, tapered_mapper, ansatz, theta,
        betas=VQD_BETAS,
        restarts=VQD_RESTARTS, maxiter=VQD_MAXITER
    )

    print("\n[CH4 | basis=STO-3G | active: HOMO-1..LUMO+1 (4e,4o) | Mapper: Parity → Z₂ tapering | "
          f"Ansatz: UCCSD(HF ref; excitations='sd') | Optim: L-BFGS-B]")
    print(f"    qubits: {n_full} → {n_tap} (tapered), terms={n_terms}, RNG_SEED={RNG_SEED}")
    print(report_line("FCI(full-H)", spec_fci.E0, spec_fci.E1, spec_fci.E2))
    print(report_line("qEOM(full)",  e0_vqe, e1_qeom, e2_qeom, ref=spec_fci,
                      extra=(f"kept_ops={qeom_info.get('kept_ops','?')}/{qeom_info.get('total_ops','?')}, "
                             f"block_dim={qeom_info.get('block_dim','?')}, "
                             f"S_rank={qeom_info.get('S_rank','?')}, "
                             f"λ_min/max(S)≈{qeom_info.get('lam_min',float('nan')):.2e}/{qeom_info.get('lam_max',float('nan')):.2e}, "
                             f"cond(S)≈{qeom_info.get('condS',float('nan')):.2e}, "
                             f"ω_min={qeom_info.get('omega_min_cut','?')}")))
    print(report_line("VQD",         e0_vqd, e1_vqd, e2_vqd, ref=spec_fci))

    # 표 출력용 스펙(통일된 필드)
    spec_qe = Spectrum(E0=e0_vqe, E1=e1_qeom, E2=e2_qeom)
    spec_vq = Spectrum(E0=e0_vqd, E1=e1_vqd,  E2=e2_vqd)

    # 최종 표
    meta = (f"basis=STO-3G | active (4e,4o) | mapper=Parity→Z₂ tapering | qubits {n_full}→{n_tap} | "
            f"ansatz=UCCSD(HF ref; 'sd')")
    print_energy_table(spec_fci, spec_qe, spec_vq, meta=meta)

    # 플롯(옵션)
    results = {
        "CH4": {
            "FCI": spec_fci,
            "qEOM": spec_qe,
            "VQD":  spec_vq,
        }
    }
    if SHOW_PLOTS or SAVE_FIGS:
        plot_all(results, title_suffix=f"  (qubits {n_full}→{n_tap})")


if __name__ == "__main__":
    main()