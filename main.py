"""Mini NPU simulator: direct-loop MAC pattern matching without external libraries."""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


EPSILON = 1e-9
MIN_BENCHMARK_REPEATS = 10
SUPPORTED_JSON_SIZES = (5, 13, 25)


def normalize_label(value: Any) -> Optional[str]:
    """Convert accepted input labels to the two internal labels."""
    if not isinstance(value, str):
        return None
    label = value.strip().lower()
    if label in {"+", "cross"}:
        return "Cross"
    if label in {"x"}:
        return "X"
    return None


def mac_score(pattern: Sequence[Sequence[float]], filter_grid: Sequence[Sequence[float]]) -> float:
    """Calculate sum(pattern[i][j] * filter_grid[i][j]) using explicit loops."""
    return sum(
        pattern[row][column] * filter_grid[row][column]
        for row in range(len(pattern))
        for column in range(len(pattern))
    )


def classify_scores(score_cross: float, score_x: float, epsilon: float = EPSILON) -> str:
    if abs(score_cross - score_x) < epsilon:
        return "UNDECIDED"
    return "Cross" if score_cross > score_x else "X"


def generate_pattern(size: int, label: str) -> List[List[int]]:
    """Generate an odd-sized binary Cross or X pattern for the bonus task."""
    normalized = normalize_label(label)
    if size < 3 or size % 2 == 0 or normalized not in {"Cross", "X"}:
        raise ValueError("크기는 3 이상의 홀수이고 라벨은 Cross 또는 X여야 합니다.")
    center = size // 2
    return [
        [
            int((row == center or column == center) if normalized == "Cross" else (row == column or row + column == size - 1))
            for column in range(size)
        ]
        for row in range(size)
    ]


def validate_grid(grid: Any, size: int) -> Tuple[bool, str]:
    if not isinstance(grid, list) or len(grid) != size:
        return False, f"행 수가 {size}와 다릅니다."
    for row_index, row in enumerate(grid, start=1):
        if not isinstance(row, list) or len(row) != size:
            return False, f"{row_index}행의 열 수가 {size}와 다릅니다."
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in row):
            return False, f"{row_index}행에 숫자가 아닌 값이 있습니다."
    return True, ""


def parse_grid_lines(size: int, title: str) -> List[List[float]]:
    while True:
        print(f"\n{title}: {size}줄을 입력하세요. 각 줄에 {size}개의 숫자를 공백으로 구분합니다.")
        rows: List[List[float]] = []
        valid = True
        for row_number in range(1, size + 1):
            raw = input(f"{row_number}/{size}> ").strip()
            parts = raw.split()
            if len(parts) != size:
                print(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")
                valid = False
                break
            try:
                rows.append([float(part) for part in parts])
            except ValueError:
                print("입력 형식 오류: 숫자만 입력하세요.")
                valid = False
                break
        if valid:
            return rows
        print("처음 행부터 다시 입력합니다.")


def benchmark(pattern: Sequence[Sequence[float]], filter_grid: Sequence[Sequence[float]], repeats: int = MIN_BENCHMARK_REPEATS) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        mac_score(pattern, filter_grid)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms / repeats


def print_performance_table(cases: Sequence[Tuple[int, Sequence[Sequence[float]], Sequence[Sequence[float]]]]) -> None:
    print("\n[성능 분석]")
    print("크기(N×N) | 평균 시간(ms) | 연산 횟수(N²)")
    print("----------|---------------|---------------")
    for size, pattern, filter_grid in cases:
        average_ms = benchmark(pattern, filter_grid)
        print(f"{size}×{size:<7} | {average_ms:>13.6f} | {size * size:>13}")


def run_user_mode() -> None:
    filter_a = parse_grid_lines(3, "필터 A")
    filter_b = parse_grid_lines(3, "필터 B")
    pattern = parse_grid_lines(3, "입력 패턴")
    score_a = mac_score(pattern, filter_a)
    score_b = mac_score(pattern, filter_b)
    result = classify_scores(score_a, score_b)
    print("\n[사용자 입력 결과]")
    print(f"필터 A MAC 점수: {score_a:.6f}")
    print(f"필터 B MAC 점수: {score_b:.6f}")
    print(f"판정: {result}")
    print(f"연산 시간(ms): {benchmark(pattern, filter_a):.6f} (10회 평균)")
    print_performance_table([(3, pattern, filter_a)])


def run_generator_mode() -> None:
    while True:
        try:
            size = int(input("생성할 크기 N을 입력하세요 (3 이상의 홀수): ").strip())
            label = input("생성할 패턴을 입력하세요 (Cross/+ 또는 X/x): ").strip()
            grid = generate_pattern(size, label)
            break
        except ValueError as error:
            print(f"입력 오류: {error}")
    print(f"\n[자동 생성 결과: {size}×{size}, {normalize_label(label)}]")
    for row in grid:
        print(" ".join(str(value) for value in row))
    print(f"자기 패턴과의 MAC 점수: {mac_score(grid, grid):.6f}")


def extract_size(key: str) -> Optional[int]:
    match = re.fullmatch(r"size_(\d+)_\d+", key)
    return int(match.group(1)) if match else None


def analyze_json(path: Path) -> None:
    print(f"\n[data.json 분석: {path}]")
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"파일 로드 FAIL: {error}")
        return

    filters = data.get("filters") if isinstance(data, dict) else None
    patterns = data.get("patterns") if isinstance(data, dict) else None
    if not isinstance(filters, dict) or not isinstance(patterns, dict):
        print("스키마 FAIL: filters와 patterns 객체가 필요합니다.")
        return

    all_cases: List[Tuple[int, List[List[float]], List[List[float]]]] = []
    failures: List[Tuple[str, str]] = []
    passed = 0

    for case_id, case in patterns.items():
        reason = ""
        size = extract_size(case_id) if isinstance(case_id, str) else None
        if size not in SUPPORTED_JSON_SIZES:
            reason = "케이스 키가 size_5/13/25_N 형식이 아닙니다."
        elif not isinstance(case, dict):
            reason = "케이스가 객체가 아닙니다."
        else:
            pattern = case.get("input")
            expected = normalize_label(case.get("expected"))
            if expected is None:
                reason = "expected 라벨이 +/x 규칙에 맞지 않습니다."
            else:
                valid, validation_reason = validate_grid(pattern, size)
                if not valid:
                    reason = f"패턴 크기/값 검증 실패: {validation_reason}"
                else:
                    cross_filter = filters.get(f"size_{size}", {}).get("cross") if isinstance(filters.get(f"size_{size}"), dict) else None
                    x_filter = filters.get(f"size_{size}", {}).get("x") if isinstance(filters.get(f"size_{size}"), dict) else None
                    valid_cross, cross_reason = validate_grid(cross_filter, size)
                    valid_x, x_reason = validate_grid(x_filter, size)
                    if not valid_cross or not valid_x:
                        reason = f"필터 크기/값 검증 실패: {cross_reason or x_reason}"
                    else:
                        score_cross = mac_score(pattern, cross_filter)
                        score_x = mac_score(pattern, x_filter)
                        result = classify_scores(score_cross, score_x)
                        status = "PASS" if result == expected else "FAIL"
                        print(f"{case_id}: Cross={score_cross:.6f}, X={score_x:.6f}, 판정={result}, expected={expected}, {status}")
                        if status == "PASS":
                            passed += 1
                        else:
                            reason = f"판정 불일치: expected={expected}, actual={result}"
                        all_cases.append((size, pattern, cross_filter))
        if reason:
            failures.append((str(case_id), reason))
            print(f"{case_id}: FAIL - {reason}")

    performance_cases: List[Tuple[int, Sequence[Sequence[float]], Sequence[Sequence[float]]]] = []
    for size in SUPPORTED_JSON_SIZES:
        filter_set = filters.get(f"size_{size}")
        pattern_case = next((case for case_id, case in patterns.items() if extract_size(str(case_id)) == size and isinstance(case, dict) and isinstance(case.get("input"), list)), None)
        if isinstance(filter_set, dict) and isinstance(pattern_case, dict):
            pattern = pattern_case["input"]
            filter_grid = filter_set.get("cross")
            if validate_grid(pattern, size)[0] and validate_grid(filter_grid, size)[0]:
                performance_cases.append((size, pattern, filter_grid))
    if performance_cases:
        print_performance_table(performance_cases)

    total = len(patterns)
    print("\n[결과 리포트]")
    print(f"전체 테스트 수: {total}")
    print(f"통과 수: {passed}")
    print(f"실패 수: {len(failures)}")
    if failures:
        print("실패 케이스 목록:")
        for case_id, reason in failures:
            print(f"- {case_id}: {reason}")
    else:
        print("실패 케이스 목록: 없음")


def main() -> None:
    print("=== Mini NPU MAC 시뮬레이터 ===")
    print("1. 사용자 입력(3×3)")
    print("2. data.json 분석")
    print("3. 보너스: Cross/X 패턴 자동 생성")
    while True:
        choice = input("모드를 선택하세요 (1/2): ").strip()
        if choice == "1":
            run_user_mode()
            return
        if choice == "2":
            analyze_json(Path(__file__).with_name("data.json"))
            return
        if choice == "3":
            run_generator_mode()
            return
        print("입력 오류: 1 또는 2를 선택하세요.")


if __name__ == "__main__":
    main()
