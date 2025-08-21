# app.py
import streamlit as st
import chess
import random
from dataclasses import dataclass

# =========================
# 설정
# =========================
st.set_page_config(page_title="Streamlit Chess vs AI", layout="wide")

# -------------------------
# 유틸: 말(유니코드) 표시
# -------------------------
PIECE_UNICODE = {
    chess.PAWN:   ("♙", "♟"),
    chess.KNIGHT: ("♘", "♞"),
    chess.BISHOP: ("♗", "♝"),
    chess.ROOK:   ("♖", "♜"),
    chess.QUEEN:  ("♕", "♛"),
    chess.KING:   ("♔", "♚"),
}
def piece_to_unicode(piece: chess.Piece) -> str:
    if not piece:
        return " "
    white, black = PIECE_UNICODE[piece.piece_type]
    return white if piece.color == chess.WHITE else black

# -------------------------
# AI 평가 함수(재료값 + 간단 가중치)
# -------------------------
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}
INF = 10**9

def evaluate(board: chess.Board) -> int:
    # 즉시 종료 상황
    if board.is_checkmate():
        # 현재 차례가 체크메이트면 큰 손해
        return -INF
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return 0

    score = 0
    for sq, piece in board.piece_map().items():
        val = PIECE_VALUE[piece.piece_type]
        score += val if piece.color == chess.WHITE else -val

    # 가벼운 가중치: 말 이동 가능성(모빌리티)
    legal_count = board.legal_moves.count()
    # 현재 차례에게 +, 상대는 -
    score += (5 * legal_count) if board.turn == chess.WHITE else -(5 * legal_count)

    # 백 기준 점수 → 현재 차례 관점 점수로 변환
    return score if board.turn == chess.WHITE else -score

# -------------------------
# AI: 네가맥스 + 알파베타
# -------------------------
def order_moves(board: chess.Board, moves):
    # 캡처, 체크를 우선
    scored = []
    for m in moves:
        s = 0
        if board.is_capture(m):
            s += 1000
        board.push(m)
        if board.is_check():
            s += 500
        board.pop()
        scored.append((s, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]

def negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    max_eval = -INF
    for move in order_moves(board, list(board.legal_moves)):
        board.push(move)
        val = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        if val > max_eval:
            max_eval = val
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break
    return max_eval

def select_ai_move(board: chess.Board, depth: int, randomness: float = 0.0) -> chess.Move:
    """randomness ∈ [0,1]. 0이면 최선 수, 0.3이면 상위 후보 중 약간 랜덤."""
    best_moves = []
    best_val = -INF
    for move in order_moves(board, list(board.legal_moves)):
        board.push(move)
        val = -negamax(board, depth - 1, -INF, INF)
        board.pop()
        if val > best_val:
            best_val = val
            best_moves = [move]
        elif val == best_val:
            best_moves.append(move)

    # 약간의 랜덤성: 동일 점수 후보에서 선택, 또는 상위권 섞기
    if best_moves and randomness > 0:
        if random.random() < randomness:
            return random.choice(best_moves)
    return random.choice(best_moves) if best_moves else random.choice(list(board.legal_moves))

# -------------------------
# 세션 상태 데이터 모델
# -------------------------
@dataclass
class GameState:
    page: str = "setup"              # "setup" | "play"
    user_white: bool = True
    difficulty: str = "보통"          # 쉬움/보통/어려움
    fen: str = chess.STARTING_FEN
    selected_sq: int | None = None
    history: list = None
    game_over_text: str | None = None
    last_move: chess.Move | None = None

def get_state() -> GameState:
    if "state" not in st.session_state:
        st.session_state.state = GameState(history=[])
    return st.session_state.state

def reset_game(user_white: bool, difficulty: str):
    st.session_state.state = GameState(
        page="play",
        user_white=user_white,
        difficulty=difficulty,
        fen=chess.STARTING_FEN,
        selected_sq=None,
        history=[],
        game_over_text=None,
        last_move=None,
    )
    st.rerun()

# -------------------------
# 난이도 매핑
# -------------------------
def difficulty_to_params(diff: str):
    # (depth, randomness)
    if diff.startswith("쉬움"):
        return 1, 0.35
    if diff.startswith("어려움"):
        return 3, 0.05
    return 2, 0.15  # 보통

# -------------------------
# 게임 종료 체크
# -------------------------
def check_game_over(board: chess.Board):
    if not board.is_game_over():
        return None
    outcome = board.outcome()
    if outcome is None:
        return "무승부"
    if outcome.winner is None:
        return "무승부 (Stalemate/Threefold/50-move)"
    return "백 승리" if outcome.winner == chess.WHITE else "흑 승리"

# -------------------------
# 사용자 인터랙션: 클릭 이동
# -------------------------
def try_user_click(sq: int, state: GameState):
    board = chess.Board(state.fen)

    # 내 차례가 아니면 무시
    user_turn_color = chess.WHITE if state.user_white else chess.BLACK
    if board.turn != user_turn_color or state.game_over_text:
        return

    piece = board.piece_at(sq)
    # 선택 없음 → 내 말 클릭 시 선택
    if state.selected_sq is None:
        if piece and piece.color == user_turn_color:
            state.selected_sq = sq
        return

    # 같은 칸 재클릭 → 선택 해제
    if state.selected_sq == sq:
        state.selected_sq = None
        return

    # 이동 시도
    move = chess.Move(state.selected_sq, sq)

    # 승격 자동 퀸(간단화)
    if board.piece_at(state.selected_sq) and board.piece_at(state.selected_sq).piece_type == chess.PAWN:
        rank_to = chess.square_rank(sq)
        if (state.user_white and rank_to == 7) or ((not state.user_white) and rank_to == 0):
            move = chess.Move(state.selected_sq, sq, promotion=chess.QUEEN)

    if move in board.legal_moves:
        # 사용자 수 진행
        san = board.san(move)
        board.push(move)
        state.last_move = move
        state.history.append(f"사용자: {san}")

        # 게임 종료 확인
        over = check_game_over(board)
        if over:
            state.game_over_text = over
            state.fen = board.fen()
            state.selected_sq = None
            return

        # AI 수 진행
        depth, rnd = difficulty_to_params(state.difficulty)
        with st.spinner("AI가 수를 생각 중..."):
            ai_move = select_ai_move(board, depth=depth, randomness=rnd)
        ai_san = board.san(ai_move)
        board.push(ai_move)
        state.last_move = ai_move
        state.history.append(f"AI: {ai_san}")

        over = check_game_over(board)
        if over:
            state.game_over_text = over

        # 상태 반영
        state.fen = board.fen()
        state.selected_sq = None
    else:
        # 불법 수 → 선택을 유지하여 다시 시도 가능
        pass

# -------------------------
# 보드 렌더링 (버튼 그리드)
# -------------------------
def render_board(state: GameState):
    board = chess.Board(state.fen)
    user_white = state.user_white
    rank_range = range(7, -1, -1) if user_white else range(0, 8)
    file_range = range(0, 8) if user_white else range(7, -1, -1)

    # 보드 상단 정보
    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        turn_text = "백(사용자)" if (board.turn == chess.WHITE and user_white) else \
                    "흑(사용자)" if (board.turn == chess.BLACK and not user_white) else "AI"
        st.markdown(f"**현재 차례:** {turn_text}")

    # 좌표 라벨 행
    files_label = [chr(ord('a') + f) for f in file_range]
    st.markdown(" ".join([f"`{f}`".center(3) for f in files_label]))

    for r in rank_range:
        cols = st.columns(8, gap="small")
        for i, f in enumerate(file_range):
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            label = piece_to_unicode(piece)
            # 선택 칸 시각 차별화: 대괄호
            if state.selected_sq == sq:
                label = f"[{label}]"

            # 고유 키
            key = f"square_{sq}_{len(state.history)}"
            if cols[i].button(label, key=key, use_container_width=True):
                try_user_click(sq, state)

        # 랭크 라벨 표시
        st.write(f"`{r+1}`")

    # 마지막 수/상태
    left, right = st.columns([2, 1])
    with left:
        st.subheader("기록")
        if state.history:
            st.write("\n".join(state.history[-12:]))
        else:
            st.write("기록 없음")

    with right:
        if state.game_over_text:
            st.error(f"게임 종료: {state.game_over_text}")
        if st.button("↩️ 한 수 무르기(사용자+AI)", disabled=len(chess.Board(state.fen).move_stack) < 2):
            board = chess.Board(state.fen)
            if len(board.move_stack) >= 2:
                # AI 수, 사용자 수 순으로 되돌리기
                board.pop()
                board.pop()
                state.fen = board.fen()
                if state.history:
                    state.history.pop()
                if state.history:
                    state.history.pop()
                state.game_over_text = None
                state.selected_sq = None
                st.rerun()

# =========================
# UI
# =========================
state = get_state()

with st.sidebar:
    st.header("새 게임")
    side = st.radio("플레이 색상", ["백(White)", "흑(Black)"], index=0 if state.user_white else 1)
    diff = st.selectbox("난이도", ["쉬움", "보통", "어려움"], index=["쉬움","보통","어려움"].index(state.difficulty))
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("새 게임 시작"):
            reset_game(user_white=(side.startswith("백")), difficulty=diff)
    with col_b:
        if st.button("도움말 보기"):
            st.session_state.show_help = True

# 도움말
with st.expander("사용 방법", expanded=st.session_state.get("show_help", False)):
    st.markdown(
        """
- 말이 있는 칸(내 말)을 **한 번 클릭** → 선택  
- 이동하려는 칸을 **다시 클릭** → 이동 시도  
- 폰 승격은 자동으로 **퀸**으로 처리  
- **한 수 무르기** 버튼은 사용자와 AI의 최근 수를 동시에 되돌립니다  
- 난이도는 탐색 깊이(쉬움=1, 보통=2, 어려움=3)와 약간의 랜덤성을 조정합니다  
        """
    )
    if st.session_state.get("show_help", False):
        st.session_state.show_help = False

# 페이지 전환: setup → play (최초)
if state.page == "setup":
    # 최초 진입 시 기본값으로 즉시 시작
    reset_game(user_white=state.user_white, difficulty=state.difficulty)

# 플레이 화면
if state.page == "play":
    # AI 선수(사용자가 흑)일 때 초기 한 수 둠
    board = chess.Board(state.fen)
    if (not state.user_white) and len(board.move_stack) == 0:
        depth, rnd = difficulty_to_params(state.difficulty)
        with st.spinner("AI가 첫 수를 두는 중..."):
            ai_move = select_ai_move(board, depth=depth, randomness=rnd)
        ai_san = board.san(ai_move)
        board.push(ai_move)
        state.fen = board.fen()
        state.history.append(f"AI: {ai_san}")

    st.title("♟️ 체스 (AI 대전)")
    render_board(state)
