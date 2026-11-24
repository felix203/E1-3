"""
Mini NPU Simulator
==================
MAC(Multiply-Accumulate) 연산으로 입력 패턴이 어떤 필터에 더 가까운지 판정하는 콘솔 프로그램.

실행: python main.py   (data.json은 main.py와 같은 폴더에 둔다)
외부 라이브러리 없이 표준 라이브러리만 사용한다. (Python 3.8 이상)
"""

import json
import math
import os
import re
import sys
import time

# ──────────────────────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────────────────────
EPSILON = 1e-9      # |점수 차이| < EPSILON 이면 동점
REPEAT = 10         # 성능 측정 반복 횟수 (MAC 연산마다 10회)
MEASURE_LABEL = f"MAC 1회 평균, 필터별 {REPEAT}회 측정"

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

# 프로그램 내부 표준 라벨
LABEL_CROSS = "Cross"
LABEL_X = "X"
UNDECIDED = "UNDECIDED"

# 라벨 정규화 규칙 (명세에 정의된 표기만 사용)
EXPECTED_TO_LABEL = {"+": LABEL_CROSS, "x": LABEL_X}        # expected 값
FILTER_KEY_TO_LABEL = {"cross": LABEL_CROSS, "x": LABEL_X}  # filter 키

FILTER_KEY_RE = re.compile(r"^size_(\d+)$")          # 예: size_13
PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(.+)$")    # 예: size_13_2 (N=13, idx=2)

# 3×3 성능 측정에 사용하는 명세 예시 데이터 (십자가 / X)
SPEC_CROSS_3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
SPEC_X_3 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


# ──────────────────────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────────────────────
class Matrix:
    """n×n 2차원 배열. set으로 특정 위치에 값을 저장하고 get으로 읽는다."""

    def __init__(self, n):
        self.size = n
        self.data = [[0.0] * n for _ in range(n)]

    def get(self, row, col):
        return self.data[row][col]

    def set(self, row, col, value):
        self.data[row][col] = float(value)

    @classmethod
    def from_rows(cls, rows, n):
        """n×n 숫자 2차원 리스트 → Matrix. 행 수/열 수/숫자 여부가 맞지 않으면 ValueError."""
        if not isinstance(rows, list):
            raise ValueError("2차원 배열이 아닙니다")
        if len(rows) != n:
            raise ValueError(f"행 수 불일치: {len(rows)}행 (필요: {n}행)")
        matrix = cls(n)
        for r, row in enumerate(rows):
            if not isinstance(row, list):
                raise ValueError(f"{r + 1}행이 배열이 아닙니다")
            if len(row) != n:
                raise ValueError(f"열 수 불일치: {r + 1}행이 {len(row)}열 (필요: {n}열)")
            for c, value in enumerate(row):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"숫자가 아닌 값: {r + 1}행 {c + 1}열 = {value!r}")
                matrix.set(r, c, value)
        return matrix


def size_text(n):
    return f"{n}×{n}"


def square_size(rows):
    """rows가 정사각 2차원 리스트면 한 변의 길이, 아니면 None."""
    if isinstance(rows, list) and rows and all(isinstance(r, list) and len(r) == len(rows) for r in rows):
        return len(rows)
    return None


# ──────────────────────────────────────────────────────────────
# MAC 연산 · 판정 · 시간 측정
# ──────────────────────────────────────────────────────────────
def mac(pattern, filt):
    """같은 위치끼리 곱하고(Multiply) 모두 더한다(Accumulate). 곱셈·덧셈 N²번 → O(N²)."""
    if pattern.size != filt.size:
        raise ValueError(f"크기 불일치: 패턴 {size_text(pattern.size)}, 필터 {size_text(filt.size)}")
    n = pattern.size
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += pattern.get(i, j) * filt.get(i, j)
    return total


def decide(score_a, score_b, label_a, label_b):
    """|A-B| < EPSILON 이면 동점(UNDECIDED), 아니면 점수가 큰 쪽."""
    if abs(score_a - score_b) < EPSILON:
        return UNDECIDED
    return label_a if score_a > score_b else label_b


def measure_mac_ms(pattern, filters):
    """각 필터와의 MAC 연산을 REPEAT회씩 한 번씩 따로 재서, MAC 1회 평균 시간(ms)을 돌려준다.

    입력/출력/파일 읽기는 포함하지 않고 연산 함수 호출 구간만 잰다.
    """
    times = []
    for filt in filters:
        for _ in range(REPEAT):
            start = time.perf_counter()
            mac(pattern, filt)
            times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)


def print_perf_table(rows):
    """rows: [(N, 평균 ms), ...]"""
    print("크기(N×N)      평균 시간(ms)   연산 횟수(N²)")
    print("-" * 44)
    for n, ms in rows:
        print(f"{size_text(n):<12}{ms:>16.4f}{n * n:>16}")


def print_section(title):
    print()
    print("#" + "-" * 40)
    print(f"# {title}")
    print("#" + "-" * 40)


# ──────────────────────────────────────────────────────────────
# 모드 1: 사용자 입력 (3×3)
# ──────────────────────────────────────────────────────────────
def parse_lines(lines, n):
    """입력받은 n줄 → (Matrix, 오류 목록). 오류가 있으면 Matrix는 None."""
    rows = []
    errors = []
    for idx, line in enumerate(lines, start=1):
        tokens = line.split()
        if not tokens:
            errors.append(f"행 수 불일치: {idx}번째 줄이 비어 있습니다 ({n}행 필요)")
            continue
        if len(tokens) != n:
            errors.append(f"열 수 불일치: {idx}번째 줄의 숫자가 {len(tokens)}개입니다 ({n}개 필요)")
            continue
        try:
            row = [float(token) for token in tokens]
        except ValueError:
            errors.append(f"숫자 파싱 실패: {idx}번째 줄 '{line.strip()}'")
            continue
        if not all(math.isfinite(v) for v in row):
            errors.append(f"숫자 파싱 실패: {idx}번째 줄 '{line.strip()}'")
            continue
        rows.append(row)
    if errors:
        return None, errors
    return Matrix.from_rows(rows, n), []


def input_matrix(title, n):
    """한 줄씩(공백 구분) n줄을 입력받는다. 형식 오류면 안내 후 다시 입력받는다."""
    while True:
        print(f"{title} ({n}줄 입력, 공백 구분)")
        lines = [input() for _ in range(n)]
        matrix, errors = parse_lines(lines, n)
        if matrix is not None:
            return matrix
        print(f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요.")
        for error in errors:
            print(f"  - {error}")
        print("다시 입력하세요.\n")


def print_matrix(matrix):
    for i in range(matrix.size):
        print("  " + " ".join(f"{matrix.get(i, j):g}" for j in range(matrix.size)))


def run_user_mode():
    n = 3

    print_section("[1] 필터 입력")
    filter_a = input_matrix("필터 A", n)
    print()
    filter_b = input_matrix("필터 B", n)

    print("\n✓ 필터 저장 완료")
    print("필터 A:")
    print_matrix(filter_a)
    print("필터 B:")
    print_matrix(filter_b)

    print_section("[2] 패턴 입력")
    pattern = input_matrix("패턴", n)

    print_section("[3] MAC 결과")
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    elapsed_ms = measure_mac_ms(pattern, [filter_a, filter_b])
    verdict = decide(score_a, score_b, "A", "B")

    print(f"A 점수: {score_a!r}")
    print(f"B 점수: {score_b!r}")
    print(f"연산 시간({MEASURE_LABEL}): {elapsed_ms:.4f} ms")
    if verdict == UNDECIDED:
        print("판정: 판정 불가 (|A-B| < 1e-9)")
    else:
        print(f"판정: {verdict}")

    print_section(f"[4] 성능 분석 ({MEASURE_LABEL})")
    print_perf_table([(n, elapsed_ms)])


# ──────────────────────────────────────────────────────────────
# 모드 2: data.json 분석
# ──────────────────────────────────────────────────────────────
def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"✗ 파일을 찾을 수 없습니다: {path}")
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        print(f"✗ JSON 형식 오류: {err}")
        return None
    except OSError as err:
        print(f"✗ 파일을 읽을 수 없습니다: {err}")
        return None
    if not isinstance(data, dict):
        print("✗ JSON 최상위 구조가 객체가 아닙니다.")
        return None
    return data


def load_filter_group(key, n, group):
    """size_N 항목 → {'Cross': Matrix, 'X': Matrix}. 문제가 있으면 ValueError."""
    if not isinstance(group, dict):
        raise ValueError("필터 객체가 아닙니다")
    result = {}
    for filter_key, rows in group.items():
        label = FILTER_KEY_TO_LABEL.get(filter_key)
        if label is None:
            raise ValueError(f"알 수 없는 필터 키 '{filter_key}' (허용: 'cross', 'x')")
        try:
            result[label] = Matrix.from_rows(rows, n)
        except ValueError as err:
            raise ValueError(f"'{filter_key}' 필터가 {key}({size_text(n)})와 맞지 않음 - {err}") from None
    for label in (LABEL_CROSS, LABEL_X):
        if label not in result:
            raise ValueError(f"{label} 필터가 없습니다")
    return result


def load_filters(raw_filters):
    """반환: ({N: {'Cross': Matrix, 'X': Matrix}}, {N: 오류 메시지})"""
    filters = {}
    errors = {}
    if not isinstance(raw_filters, dict):
        print("✗ filters 항목이 없거나 형식이 잘못되었습니다.")
        return filters, errors
    for key, group in raw_filters.items():
        match = FILTER_KEY_RE.match(key)
        if not match:
            print(f"✗ {key}: 키 형식 오류 (size_N 형식이 아님)")
            continue
        n = int(match.group(1))
        try:
            filters[n] = load_filter_group(key, n, group)
            print(f"✓ {key:<7} 필터 로드 완료 ({LABEL_CROSS}, {LABEL_X})")
        except ValueError as err:
            errors[n] = f"{key} 필터 로드 실패: {err}"
            print(f"✗ {errors[n]}")
    return filters, errors


def analyze_case(key, case, filters, filter_errors):
    """패턴 1건을 검증 → MAC 연산 → 판정. 결과 dict를 돌려준다."""
    result = {"name": key, "passed": False, "reason": "", "size": None, "pattern": None}

    def fail(reason):
        print(f"FAIL ({reason})")
        result["reason"] = reason
        return result

    # 키에서 N을 추출해 size_N 필터 선택
    match = PATTERN_KEY_RE.match(key)
    if not match:
        return fail("키 형식 오류: size_{N}_{idx} 형식이 아님 (N은 숫자)")
    n = int(match.group(1))
    if n in filter_errors:
        return fail(filter_errors[n])
    if n not in filters:
        return fail(f"size_{n} 필터가 없음")
    if not isinstance(case, dict):
        return fail("패턴 항목 형식 오류")

    # 패턴 로드 및 필터와 크기 비교
    rows = case.get("input")
    filter_size = filters[n][LABEL_CROSS].size
    input_size = square_size(rows)
    if input_size is not None and input_size != filter_size:
        return fail(f"크기 불일치: 패턴 {size_text(input_size)}, 필터 {size_text(filter_size)}")
    try:
        pattern = Matrix.from_rows(rows, filter_size)
    except ValueError as err:
        return fail(f"input 오류 (필터 {size_text(filter_size)} 기준): {err}")
    result["size"] = pattern.size
    result["pattern"] = pattern

    # expected 라벨 정규화
    raw_expected = case.get("expected")
    expected = EXPECTED_TO_LABEL.get(raw_expected) if isinstance(raw_expected, str) else None
    if expected is None:
        return fail(f"expected 값 오류: {raw_expected!r} (허용: '+', 'x')")

    # MAC 연산 및 판정
    score_cross = mac(pattern, filters[n][LABEL_CROSS])
    score_x = mac(pattern, filters[n][LABEL_X])
    verdict = decide(score_cross, score_x, LABEL_CROSS, LABEL_X)

    print(f"Cross 점수: {score_cross!r}")
    print(f"X 점수: {score_x!r}")
    if verdict == expected:
        print(f"판정: {verdict} | expected: {expected} | PASS")
        result["passed"] = True
    elif verdict == UNDECIDED:
        print(f"판정: {verdict} | expected: {expected} | FAIL (동점 규칙)")
        result["reason"] = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
    else:
        print(f"판정: {verdict} | expected: {expected} | FAIL")
        result["reason"] = f"판정({verdict})과 expected({expected}) 불일치"
    return result


def analyze_patterns(raw_patterns, filters, filter_errors):
    if not isinstance(raw_patterns, dict):
        print("✗ patterns 항목이 없거나 형식이 잘못되었습니다.")
        return []
    results = []
    for key, case in raw_patterns.items():
        print(f"--- {key} ---")
        try:
            results.append(analyze_case(key, case, filters, filter_errors))
        except Exception as err:  # 어떤 오류도 해당 케이스 FAIL로 처리하고 다음 케이스로 진행
            reason = f"처리 중 오류: {err}"
            print(f"FAIL ({reason})")
            results.append({"name": key, "passed": False, "reason": reason, "size": None, "pattern": None})
    return results


def run_performance(filters, results):
    """3×3(명세 예시)과 data.json의 5×5/13×13/25×25로 크기별 MAC 평균 시간을 잰다."""
    rows = []
    notes = []

    spec_cross = Matrix.from_rows(SPEC_CROSS_3, 3)
    spec_x = Matrix.from_rows(SPEC_X_3, 3)
    rows.append((3, measure_mac_ms(spec_cross, [spec_cross, spec_x])))

    for n in (5, 13, 25):
        pattern = next((r["pattern"] for r in results if r["size"] == n), None)
        if n not in filters or pattern is None:
            notes.append(f"{size_text(n)}: data.json에 사용할 필터/패턴이 없어 측정하지 못했습니다.")
            continue
        rows.append((n, measure_mac_ms(pattern, [filters[n][LABEL_CROSS], filters[n][LABEL_X]])))

    print_perf_table(rows)
    for note in notes:
        print(f"※ {note}")


def print_summary(results):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {total - passed}개")
    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\n실패 케이스:")
        for r in failures:
            print(f"- {r['name']}: {r['reason']}")
    print('\n(상세 원인 분석 및 복잡도 설명은 README.md의 "결과 리포트" 섹션에 작성)')


def run_json_mode():
    print_section("[1] 필터 로드")
    data = load_json(DATA_FILE)
    if data is None:
        print("data.json을 읽지 못해 분석을 종료합니다.")
        return
    filters, filter_errors = load_filters(data.get("filters"))

    print_section("[2] 패턴 분석 (라벨 정규화 적용)")
    results = analyze_patterns(data.get("patterns"), filters, filter_errors)

    print_section(f"[3] 성능 분석 ({MEASURE_LABEL})")
    run_performance(filters, results)

    print_section("[4] 결과 요약")
    print_summary(results)


# ──────────────────────────────────────────────────────────────
# 시작점
# ──────────────────────────────────────────────────────────────
def main():
    # 콘솔이 표시할 수 없는 기호(✓ 등)가 있어도 프로그램이 멈추지 않게 한다
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except (ValueError, OSError):
            pass

    print("=== Mini NPU Simulator ===\n")
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    try:
        while True:
            choice = input("선택: ").strip()
            if choice in ("1", "2"):
                break
            print("1 또는 2를 입력하세요.")
        if choice == "1":
            run_user_mode()
        else:
            run_json_mode()
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 중단되어 프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
