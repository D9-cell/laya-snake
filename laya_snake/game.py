import random
from collections import deque

DIRS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


class SnakeGame:
    def __init__(self, width=24, height=14):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        cx, cy = self.width // 2, self.height // 2
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.direction = "right"
        self.score = 0
        self.game_over = False
        self.food = self._spawn_food()

    def head(self):
        return self.snake[0]

    def _spawn_food(self):
        occupied = set(self.snake)
        free = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]
        return random.choice(free) if free else None

    def _next_head(self, direction):
        dx, dy = DIRS[direction]
        hx, hy = self.head()
        return hx + dx, hy + dy

    def is_fatal(self, direction):
        nx, ny = self._next_head(direction)
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            return True
        ate = (nx, ny) == self.food
        blocked = set(self.snake) if ate else set(list(self.snake)[:-1])
        return (nx, ny) in blocked

    def step(self, direction):
        if self.game_over:
            return
        nx, ny = self._next_head(direction)
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            self.game_over = True
            self.direction = direction
            return
        new_head = (nx, ny)
        ate = new_head == self.food
        blocked = set(self.snake) if ate else set(list(self.snake)[:-1])
        if new_head in blocked:
            self.game_over = True
            self.direction = direction
            return
        self.snake.appendleft(new_head)
        if ate:
            self.score += 1
            self.food = self._spawn_food()
        else:
            self.snake.pop()
        self.direction = direction

    def board_fill_ratio(self):
        return len(self.snake) / (self.width * self.height)


class GameSnapshot:
    """Immutable copy of the fields LayaBrain.decide() reads, so the
    background inference thread never touches the live, mutable game."""

    def __init__(self, game):
        self.width = game.width
        self.height = game.height
        self.direction = game.direction
        self.snake = list(game.snake)
        self.food = game.food

    def head(self):
        return self.snake[0]

    def is_fatal(self, direction):
        dx, dy = DIRS[direction]
        hx, hy = self.head()
        nx, ny = hx + dx, hy + dy
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            return True
        ate = (nx, ny) == self.food
        blocked = set(self.snake) if ate else set(self.snake[:-1])
        return (nx, ny) in blocked


def flood_fill_area(start, blocked, width, height):
    if start in blocked:
        return 0
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in blocked and (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return len(seen)


def bfs_path_exists(start, goal, blocked, width, height):
    if start == goal:
        return True
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if (nx, ny) == goal:
                return True
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in blocked and (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return False


def estimate_risk_and_reachability(game, executed_direction):
    """Heuristic (flood-fill) estimates computed from game state after a move
    would execute — NOT produced by the model. Used for the side-panel gauges."""
    nx, ny = game._next_head(executed_direction)
    new_head = (nx, ny)
    ate = new_head == game.food
    body_after = deque(game.snake)
    body_after.appendleft(new_head)
    if not ate:
        body_after.pop()
    blocked = set(list(body_after)[1:])  # everything except the new head
    area = flood_fill_area(new_head, blocked, game.width, game.height)
    dead_end_risk = max(0.0, min(1.0, 1 - area / max(1, len(body_after))))
    reachable = bfs_path_exists(new_head, game.food, blocked, game.width, game.height) if game.food else False
    return dead_end_risk, (1.0 if reachable else 0.0)
