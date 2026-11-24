"""
Mini NPU Simulator
==================
MAC(Multiply-Accumulate) 연산으로 입력 패턴이 어떤 필터에 더 가까운지 판정하는 콘솔 프로그램.

실행:        python main.py              (data.json은 main.py와 같은 폴더에 둔다)
단위 테스트: python -m doctest -v main.py
외부 라이브러리 없이 표준 라이브러리만 사용한다. (Python 3.8 이상)
"""

import datetime
import gc
import json
import math
import os
import platform
import re
import statistics
import sys
import time
import unicodedata

# ──────────────────────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────────────────────
EPSILON = 1e-9      # |점수 차이| < EPSILON 이면 동점 (근거는 README "동점 처리 정책" 참고)
REPEAT = 10         # 성능 측정 반복 횟수 (MAC 연산마다 10회)
MEASURE_LABEL = f"MAC 1회 평균, 필터별 {REPEAT}회 측정"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")      # 모드 2 입력 파일
LOG_FILE = os.path.join(BASE_DIR, "mini_npu.log")    # 모드 2 실패 사유 로그

# 프로그램 내부 표준 라벨. 새 라벨은 STANDARD_LABELS와 아래 두 매핑에 추가한다.
LABEL_CROSS = "Cross"
LABEL_X = "X"
STANDARD_LABELS = (LABEL_CROSS, LABEL_X)
UNDECIDED = "UNDECIDED"

# 라벨 정규화 규칙: 외부 표기 → 표준 라벨 (명세에 정의된 표기)
EXPECTED_TO_LABEL = {"+": LABEL_CROSS, "x": LABEL_X}        # patterns의 expected 값
FILTER_KEY_TO_LABEL = {"cross": LABEL_CROSS, "x": LABEL_X}  # filters의 키
LABEL_RULES = {"expected": EXPECTED_TO_LABEL, "filter": FILTER_KEY_TO_LABEL}

FILTER_KEY_RE = re.compile(r"^size_(\d+)$")          # 예: size_13
PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(.+)$")    # 예: size_13_2 (N=13, idx=2)

# 3×3 성능 측정에 사용하는 명세 예시 데이터 (Case 1·2의 십자가 / X)
SPEC_CROSS_3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
SPEC_X_3 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


# ──────────────────────────────────────────────────────────────
# 라벨 정규화
# ──────────────────────────────────────────────────────────────
def normalize_label(raw, kind):
    """외부 표기를 표준 라벨로 바꾼다. 규칙에 없는 표기면 None.

    kind: "expected"(patterns의 expected 값) 또는 "filter"(filters의 키)

    >>> normalize_label("+", "expected")
    'Cross'
    >>> normalize_label("x", "expected")
    'X'
    >>> normalize_label("cross", "filter")
    'Cross'
    >>> normalize_label("x", "filter")
    'X'
    >>> print(normalize_label("X", "expected"))       # 규칙에 없는 표기
    None
    >>> print(normalize_label("cross", "expected"))   # filter 키 표기는 expected 값으로 인정하지 않음
    None
    >>> print(normalize_label(None, "expected"))      # 문자열이 아님
    None
    """
    if not isinstance(raw, str):
        return None
    return LABEL_RULES[kind].get(raw)


# ──────────────────────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────────────────────
class Matrix:
    """n×n 2차원 배열. set으로 특정 위치에 값을 저장하고 get으로 읽는다.

    >>> m = Matrix(3)
    >>> m.set(1, 2, 7)
    >>> m.get(1, 2)
    7.0
    """

    def __init__(self, n):
        self.size = n
        self.data = [[0.0] * n for _ in range(n)]

    def get(self, row, col):
        return self.data[row][col]

    def set(self, row, col, value):
        self.data[row][col] = float(value)

    @classmethod
    def from_rows(cls, rows, n):
        """n×n 숫자 2차원 리스트 → Matrix. 행 수/열 수/숫자 여부가 맞지 않으면 ValueError.

        >>> Matrix.from_rows([[1, 0], [0, 1]], 2).get(1, 1)
        1.0
        >>> Matrix.from_rows([[1, 0], [0]], 2)
        Traceback (most recent call last):
        ...
        ValueError: 열 수 불일치: 2행이 1열 (필요: 2열)
        """
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


def square_size(rows):
    """rows가 정사각 2차원 리스트면 한 변의 길이, 아니면 None."""
    if isinstance(rows, list) and rows and all(isinstance(r, list) and len(r) == len(rows) for r in rows):
        return len(rows)
    return None


def size_text(n):
    return f"{n}×{n}"


def sci(value):
    """1e-09 대신 1e-9처럼 짧은 지수 표기."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


# ──────────────────────────────────────────────────────────────
# MAC 연산 · 판정
# ──────────────────────────────────────────────────────────────
def mac(pattern, filt):
    """같은 위치끼리 곱하고(Multiply) 모두 더한다(Accumulate).

    N×N이면 칸마다 곱셈 1번 + 덧셈 1번 → 곱셈 N²번, 덧셈 N²번 → O(N²).
    예) 3×3: 곱셈 9번·덧셈 9번, 25×25: 곱셈 625번·덧셈 625번

    >>> cross = Matrix.from_rows(SPEC_CROSS_3, 3)
    >>> x = Matrix.from_rows(SPEC_X_3, 3)
    >>> mac(cross, cross)   # 명세 Case 1
    5.0
    >>> mac(cross, x)       # 명세 Case 2
    1.0
    """
    if pattern.size != filt.size:
        raise ValueError(f"크기 불일치: 패턴 {size_text(pattern.size)}, 필터 {size_text(filt.size)}")
    n = pattern.size
    total = 0.0
    for i in range(n):              # 행 우선(row-major) 순서로 접근
        for j in range(n):
            total += pattern.get(i, j) * filt.get(i, j)
    return total


def decide(scores):
    """{라벨: 점수}에서 점수가 가장 큰 라벨을 고른다. 1등과 2등의 차이가 EPSILON 미만이면 UNDECIDED.

    >>> decide({"A": 1.0, "B": 5.0})
    'B'
    >>> s = 0.0
    >>> for _ in range(9):   # MAC처럼 0.1을 차례로 9번 누적
    ...     s += 0.1
    >>> s
    0.8999999999999999
    >>> decide({"Cross": 0.9, "X": s})
    'UNDECIDED'
    """
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < EPSILON:
        return UNDECIDED
    return ranked[0][0]


def tie_detail(scores):
    """동점 근거 문자열.

    >>> tie_detail({"Cross": 0.9, "X": 0.8999999999999999})
    '|Cross-X| = 1.11e-16 < 1e-9'
    """
    top_two = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:2]
    order = list(scores)
    (a, score_a), (b, score_b) = sorted(top_two, key=lambda item: order.index(item[0]))
    return f"|{a}-{b}| = {abs(score_a - score_b):.2e} < {sci(EPSILON)}"


def scores_text(scores):
    return ", ".join(f"{label} {score!r}" for label, score in scores.items())


# ──────────────────────────────────────────────────────────────
# 성능 측정
# ──────────────────────────────────────────────────────────────
def measure_mac_times(pattern, filters):
    """각 필터와의 MAC 연산을 REPEAT회씩, 한 번씩 따로 잰다. 반환: 측정값(ms) 목록.

    - 입력/출력/파일 읽기는 포함하지 않고 연산 함수 호출 구간만 잰다.
    - 측정 동안 가비지 컬렉션(GC)을 잠시 끈다. GC가 측정 도중 실행되면 그 회차만
      시간이 튀어 평균이 흔들리기 때문이다. (표준 라이브러리 timeit도 같은 방식)
    """
    times = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for filt in filters:
            for _ in range(REPEAT):
                start = time.perf_counter()
                mac(pattern, filt)
                times.append((time.perf_counter() - start) * 1000)
    finally:
        if gc_was_enabled:
            gc.enable()
    return times


def summarize(times):
    """측정값(ms) 목록 → (평균, 최소, 최대, 표준편차)

    >>> summarize([1.0, 2.0, 3.0])
    (2.0, 1.0, 3.0, 1.0)
    """
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    return statistics.mean(times), min(times), max(times), stdev


def environment_text():
    """측정 환경: Python 버전 / OS / CPU / 논리 코어 수"""
    cpu = platform.processor() or platform.machine()
    return (f"Python {platform.python_version()} / {platform.system()} {platform.release()} / "
            f"{cpu} / 논리 코어 {os.cpu_count()}개")


# ──────────────────────────────────────────────────────────────
# 출력 도우미
# ──────────────────────────────────────────────────────────────
def display_width(text):
    """콘솔에서 보이는 폭. 한글처럼 넓은 글자는 2칸으로 센다."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def table_row(cells, widths):
    """첫 칸은 왼쪽, 나머지는 오른쪽 정렬."""
    out = []
    for i, (cell, width) in enumerate(zip(cells, widths)):
        text = str(cell)
        gap = " " * max(0, width - display_width(text))
        out.append(text + gap if i == 0 else gap + text)
    return "".join(out)


def print_perf_table(rows):
    """rows: [(N, 측정값 목록), ...] → 크기(N×N) / 평균 시간(ms) / 연산 횟수(N²) / 최소 / 최대 / 표준편차"""
    headers = ["크기(N×N)", "평균 시간(ms)", "연산 횟수(N²)", "최소(ms)", "최대(ms)", "표준편차(ms)"]
    widths = [11, 15, 15, 11, 11, 14]
    print(table_row(headers, widths))
    print("-" * sum(widths))
    for n, times in rows:
        mean, low, high, stdev = summarize(times)
        print(table_row([size_text(n), f"{mean:.4f}", n * n, f"{low:.4f}", f"{high:.4f}", f"{stdev:.4f}"], widths))


def print_section(title):
    print()
    print("#" + "-" * 40)
    print(f"# {title}")
    print("#" + "-" * 40)


def print_matrix(matrix):
    for i in range(matrix.size):
        print("  " + " ".join(f"{matrix.get(i, j):g}" for j in range(matrix.size)))


# ──────────────────────────────────────────────────────────────
# 모드 1: 사용자 입력 (3×3)
# ──────────────────────────────────────────────────────────────
def parse_lines(lines, n):
    """입력받은 n줄 → (Matrix, 오류 목록). 오류가 있으면 Matrix는 None.

    >>> parse_lines(["0 1 0", "1 1", "0 1 0"], 3)[1]
    ['열 수 불일치: 2번째 줄의 숫자가 2개입니다 (3개 필요)']
    """
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
    """한 줄씩(공백 구분) n줄을 입력받는다. 형식 오류면 원인과 올바른 예를 보여 주고 다시 입력받는다."""
    while True:
        print(f"{title} ({n}줄 입력, 공백 구분)")
        lines = [input() for _ in range(n)]
        matrix, errors = parse_lines(lines, n)
        if matrix is not None:
            return matrix
        print(f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요.")
        for error in errors:
            print(f"  - {error}")
        print(f"  올바른 입력 예: 0 1 0  (한 줄에 숫자 {n}개, 총 {n}줄)")
        print("다시 입력하세요.\n")


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
    scores = {"A": mac(pattern, filter_a), "B": mac(pattern, filter_b)}
    times = measure_mac_times(pattern, [filter_a, filter_b])
    verdict = decide(scores)

    print(f"A 점수: {scores['A']!r}")
    print(f"B 점수: {scores['B']!r}")
    print(f"연산 시간({MEASURE_LABEL}): {statistics.mean(times):.4f} ms")
    if verdict == UNDECIDED:
        print(f"판정: 판정 불가 ({tie_detail(scores)})")
    else:
        print(f"판정: {verdict}")

    print_section(f"[4] 성능 분석 ({MEASURE_LABEL})")
    print(f"실행 환경: {environment_text()}")
    print_perf_table([(n, times)])


# ──────────────────────────────────────────────────────────────
# 모드 2: data.json 분석
# ──────────────────────────────────────────────────────────────
def load_json(path):
    """반환: (data, 오류 메시지). 성공하면 오류 메시지는 None."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, f"파일을 찾을 수 없습니다: {path}"
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        return None, f"JSON 형식 오류: {err}"
    except OSError as err:
        return None, f"파일을 읽을 수 없습니다: {err}"
    if not isinstance(data, dict):
        return None, "JSON 최상위 구조가 객체가 아닙니다."
    return data, None


def load_filter_group(key, n, group):
    """size_N 항목 → {표준 라벨: Matrix}. 문제가 있으면 ValueError."""
    if not isinstance(group, dict):
        raise ValueError("필터 객체가 아닙니다")
    result = {}
    for filter_key, rows in group.items():
        label = normalize_label(filter_key, "filter")
        if label is None:
            allowed = ", ".join(f"'{k}'" for k in FILTER_KEY_TO_LABEL)
            raise ValueError(f"알 수 없는 필터 키 '{filter_key}' (허용: {allowed})")
        try:
            result[label] = Matrix.from_rows(rows, n)
        except ValueError as err:
            raise ValueError(f"'{filter_key}' 필터가 {key}({size_text(n)})와 맞지 않음 - {err}") from None
    for label in STANDARD_LABELS:
        if label not in result:
            raise ValueError(f"{label} 필터가 없습니다")
    return result


def load_filters(raw_filters):
    """반환: ({N: {표준 라벨: Matrix}}, {N: 오류 메시지}, [로그용 실패 메시지])"""
    filters, errors, problems = {}, {}, []
    if not isinstance(raw_filters, dict):
        problems.append("filters 항목이 없거나 형식이 잘못되었습니다.")
        print(f"✗ {problems[-1]}")
        return filters, errors, problems
    for key, group in raw_filters.items():
        match = FILTER_KEY_RE.match(key)
        if not match:
            problems.append(f"키 형식 오류: '{key}' (size_N 형식이 아님, N은 숫자)")
            print(f"✗ {problems[-1]}")
            continue
        n = int(match.group(1))
        try:
            filters[n] = load_filter_group(key, n, group)
            print(f"✓ {key:<7} 필터 로드 완료 ({', '.join(STANDARD_LABELS)})")
        except ValueError as err:
            errors[n] = f"{key} 필터 로드 실패: {err}"
            problems.append(errors[n])
            print(f"✗ {errors[n]}")
    return filters, errors, problems


def new_result(key):
    return {"name": key, "passed": False, "reason": "", "detail": "", "size": None, "pattern": None}


def analyze_case(key, case, filters, filter_errors):
    """패턴 1건을 검증 → MAC 연산 → 판정. 결과 dict를 돌려준다."""
    result = new_result(key)

    def fail(reason):
        print(f"FAIL ({reason})")
        result["reason"] = reason
        return result

    # 키에서 N을 추출해 size_N 필터 선택
    match = PATTERN_KEY_RE.match(key)
    if not match:
        return fail(f"키 형식 오류: '{key}' (size_{{N}}_{{idx}} 형식이 아님, N은 숫자)")
    n = int(match.group(1))
    if n in filter_errors:
        return fail(filter_errors[n])
    if n not in filters:
        return fail(f"size_{n} 필터가 없음 (키 '{key}')")
    if not isinstance(case, dict):
        return fail("패턴 항목 형식 오류")

    # 패턴 로드 및 필터와 크기 비교
    rows = case.get("input")
    filter_size = filters[n][STANDARD_LABELS[0]].size
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
    expected = normalize_label(raw_expected, "expected")
    if expected is None:
        allowed = ", ".join(f"'{k}'" for k in EXPECTED_TO_LABEL)
        return fail(f"expected 값 오류: {raw_expected!r} (허용: {allowed})")

    # MAC 연산 및 판정
    scores = {label: mac(pattern, filters[n][label]) for label in STANDARD_LABELS}
    verdict = decide(scores)
    for label, score in scores.items():
        print(f"{label} 점수: {score!r}")

    if verdict == expected:
        print(f"판정: {verdict} | expected: {expected} | PASS")
        result["passed"] = True
    elif verdict == UNDECIDED:
        print(f"판정: {verdict} | expected: {expected} | FAIL (동점 규칙: {tie_detail(scores)})")
        result["reason"] = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
        result["detail"] = f"{scores_text(scores)} / {tie_detail(scores)}"
    else:
        print(f"판정: {verdict} | expected: {expected} | FAIL")
        result["reason"] = f"판정({verdict})과 expected({expected}) 불일치"
        result["detail"] = scores_text(scores)
    return result


def analyze_patterns(raw_patterns, filters, filter_errors):
    if not isinstance(raw_patterns, dict):
        print("✗ patterns 항목이 없거나 형식이 잘못되었습니다.")
        return [], ["patterns 항목이 없거나 형식이 잘못되었습니다."]
    results = []
    for key, case in raw_patterns.items():
        print(f"--- {key} ---")
        try:
            results.append(analyze_case(key, case, filters, filter_errors))
        except Exception as err:  # 어떤 오류도 해당 케이스 FAIL로 처리하고 다음 케이스로 진행
            result = new_result(key)
            result["reason"] = f"처리 중 오류: {err}"
            print(f"FAIL ({result['reason']})")
            results.append(result)
    return results, []


def run_performance(filters, results):
    """3×3(명세 예시)과 data.json의 5×5/13×13/25×25로 크기별 MAC 시간을 잰다."""
    rows = []
    notes = []

    spec_cross = Matrix.from_rows(SPEC_CROSS_3, 3)
    spec_x = Matrix.from_rows(SPEC_X_3, 3)
    rows.append((3, measure_mac_times(spec_cross, [spec_cross, spec_x])))

    for n in (5, 13, 25):
        pattern = next((r["pattern"] for r in results if r["size"] == n), None)
        if n not in filters or pattern is None:
            notes.append(f"{size_text(n)}: data.json에 사용할 필터/패턴이 없어 측정하지 못했습니다.")
            continue
        rows.append((n, measure_mac_times(pattern, [filters[n][label] for label in STANDARD_LABELS])))

    print(f"실행 환경: {environment_text()}")
    print_perf_table(rows)
    for note in notes:
        print(f"※ {note}")


def failure_line(result):
    line = f"- {result['name']}: {result['reason']}"
    if result["detail"]:
        line += f" [{result['detail']}]"
    return line


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
            print(failure_line(r))
    print('\n(상세 원인 분석 및 복잡도 설명은 README.md의 "결과 리포트" 섹션에 작성)')


def write_log(lines):
    """모드 2의 로드 실패·FAIL 사유를 LOG_FILE에 저장한다 (실행할 때마다 새로 씀)."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"[{stamp}] Mini NPU Simulator - data.json 분석 로그\n")
            f.write("\n".join(lines) + "\n")
    except OSError as err:
        print(f"※ 로그 파일을 저장하지 못했습니다: {err}")
        return
    print(f"(실패 사유 로그: {os.path.basename(LOG_FILE)})")


def run_json_mode():
    print_section("[1] 필터 로드")
    data, load_error = load_json(DATA_FILE)
    if data is None:
        print(f"✗ {load_error}")
        print("data.json을 읽지 못해 분석을 종료합니다.")
        write_log([f"data.json 로드 실패: {load_error}"])
        return
    filters, filter_errors, filter_problems = load_filters(data.get("filters"))

    print_section("[2] 패턴 분석 (라벨 정규화 적용)")
    results, pattern_problems = analyze_patterns(data.get("patterns"), filters, filter_errors)

    print_section(f"[3] 성능 분석 ({MEASURE_LABEL})")
    run_performance(filters, results)

    print_section("[4] 결과 요약")
    print_summary(results)

    passed = sum(1 for r in results if r["passed"])
    log = [f"총 테스트: {len(results)}개, 통과: {passed}개, 실패: {len(results) - passed}개",
           "[필터/데이터 로드 실패]"]
    log += [f"- {p}" for p in filter_problems + pattern_problems] or ["- 없음"]
    log.append("[FAIL 케이스]")
    log += [failure_line(r) for r in results if not r["passed"]] or ["- 없음"]
    write_log(log)


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
