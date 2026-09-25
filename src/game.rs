use rand::{rng, Rng};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right,
}

impl Dir {
    /// Canonical order used by the UI and the prompt: up, down, left, right.
    pub const ALL: [Dir; 4] = [Dir::Up, Dir::Down, Dir::Left, Dir::Right];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn delta(self) -> (i32, i32) {
        match self {
            Dir::Up => (0, -1),
            Dir::Down => (0, 1),
            Dir::Left => (-1, 0),
            Dir::Right => (1, 0),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Dir::Up => "up",
            Dir::Down => "down",
            Dir::Left => "left",
            Dir::Right => "right",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    fn step(self, dir: Dir) -> Point {
        let (dx, dy) = dir.delta();
        Point {
            x: self.x + dx,
            y: self.y + dy,
        }
    }
}

#[derive(Clone, Copy)]
struct Board {
    w: i32,
    h: i32,
}

impl Board {
    fn new(width: usize, height: usize) -> Self {
        Self {
            w: width as i32,
            h: height as i32,
        }
    }

    #[inline]
    fn contains(self, p: Point) -> bool {
        p.x >= 0 && p.x < self.w && p.y >= 0 && p.y < self.h
    }

    #[inline]
    fn idx(self, p: Point) -> usize {
        (p.y * self.w + p.x) as usize
    }

    fn cells(self) -> usize {
        (self.w * self.h) as usize
    }
}

/// How many foods are on the board at once, and whether rocks are scattered across it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rules {
    pub foods: usize,
    pub obstacles: bool,
}

pub const MAX_FOODS: usize = 12;

impl Default for Rules {
    fn default() -> Self {
        Self {
            foods: 5,
            obstacles: true,
        }
    }
}

/// Would moving the head to `p` end the game? `body` is head-first; the tail cell only counts when
/// the move eats (the tail stays put then). `walls` marks obstacle cells.
fn fatal_at(
    board: Board,
    walls: &[bool],
    len: usize,
    body: impl Iterator<Item = Point>,
    foods: &[Point],
    p: Point,
) -> bool {
    if !board.contains(p) || walls[board.idx(p)] {
        return true;
    }
    let ate = foods.contains(&p);
    let end = if ate { len } else { len.saturating_sub(1) };
    body.take(end).any(|x| x == p)
}

#[derive(Clone, Debug)]
pub struct SnakeGame {
    pub width: usize,
    pub height: usize,
    pub snake: VecDeque<Point>,
    pub direction: Dir,
    pub score: u32,
    pub game_over: bool,
    pub foods: Vec<Point>,
    pub obstacles: Vec<Point>,
    pub rules: Rules,
    walls: Vec<bool>,
}

impl SnakeGame {
    pub fn new(width: usize, height: usize) -> Self {
        Self::with_rules(width, height, Rules::default())
    }

    pub fn with_rules(width: usize, height: usize, rules: Rules) -> Self {
        let mut game = Self {
            width,
            height,
            snake: VecDeque::new(),
            direction: Dir::Right,
            score: 0,
            game_over: false,
            foods: Vec::new(),
            obstacles: Vec::new(),
            rules,
            walls: vec![false; width * height],
        };
        game.reset();
        game
    }

    fn board(&self) -> Board {
        Board::new(self.width, self.height)
    }

    /// A fresh round: new snake in the middle, a new rock layout and a full set of foods.
    pub fn reset(&mut self) {
        let cx = (self.width / 2) as i32;
        let cy = (self.height / 2) as i32;
        self.snake = VecDeque::from([
            Point { x: cx, y: cy },
            Point { x: cx - 1, y: cy },
            Point { x: cx - 2, y: cy },
        ]);
        self.direction = Dir::Right;
        self.score = 0;
        self.game_over = false;
        self.foods.clear();
        self.set_obstacles(if self.rules.obstacles {
            self.generate_obstacles()
        } else {
            Vec::new()
        });
        self.refill_food();
    }

    /// Replaces the obstacle layout (used by tests and `reset`).
    pub fn set_obstacles(&mut self, obstacles: Vec<Point>) {
        let board = self.board();
        self.walls = vec![false; board.cells()];
        for &p in &obstacles {
            if board.contains(p) {
                self.walls[board.idx(p)] = true;
            }
        }
        self.obstacles = obstacles;
    }

    /// Changes how many foods are kept on the board, adding or dropping foods right away.
    pub fn set_food_target(&mut self, n: usize) {
        self.rules.foods = n.clamp(1, MAX_FOODS);
        self.foods.truncate(self.rules.foods);
        self.refill_food();
    }

    pub fn head(&self) -> Point {
        self.snake[0]
    }

    pub fn is_obstacle(&self, p: Point) -> bool {
        let board = self.board();
        board.contains(p) && self.walls[board.idx(p)]
    }

    /// Rock clusters that keep every free cell connected, stay two cells off the border, keep at
    /// least two cells between clusters, and leave the spawn lane clear.
    fn generate_obstacles(&self) -> Vec<Point> {
        const SHAPES: [&[(i32, i32)]; 7] = [
            &[(0, 0)],
            &[(0, 0), (1, 0)],
            &[(0, 0), (0, 1)],
            &[(0, 0), (1, 0), (0, 1), (1, 1)],
            &[(0, 0), (1, 0), (2, 0)],
            &[(0, 0), (0, 1), (0, 2)],
            &[(0, 0), (1, 0), (0, 1)],
        ];
        let board = self.board();
        if board.w < 8 || board.h < 7 {
            return Vec::new();
        }
        let budget = board.cells() * 5 / 100;
        let (cx, cy) = (board.w / 2, board.h / 2);
        let mut r = rng();
        let mut walls = vec![false; board.cells()];
        let mut placed: Vec<Point> = Vec::new();
        let mut seen = vec![false; board.cells()];
        let mut queue = Vec::with_capacity(board.cells());
        for _ in 0..400 {
            if placed.len() >= budget {
                break;
            }
            let shape = SHAPES[r.random_range(0..SHAPES.len())];
            let ox = r.random_range(2..board.w - 2);
            let oy = r.random_range(2..board.h - 2);
            let cells: Vec<Point> = shape
                .iter()
                .map(|&(dx, dy)| Point {
                    x: ox + dx,
                    y: oy + dy,
                })
                .collect();
            let fits = cells.iter().all(|p| {
                p.x >= 2
                    && p.x < board.w - 2
                    && p.y >= 2
                    && p.y < board.h - 2
                    && !((p.y - cy).abs() <= 1 && p.x >= cx - 5 && p.x <= cx + 4)
                    && placed
                        .iter()
                        .all(|q| (q.x - p.x).abs().max((q.y - p.y).abs()) >= 3)
            });
            if !fits {
                continue;
            }
            for p in &cells {
                walls[board.idx(*p)] = true;
            }
            let start = Point { x: cx, y: cy };
            let free = board.cells() - placed.len() - cells.len();
            if flood_fill_area(board, start, &walls, &mut seen, &mut queue) == free {
                placed.extend(cells);
            } else {
                for p in &cells {
                    walls[board.idx(*p)] = false;
                }
            }
        }
        placed
    }

    /// Tops the board up to the target number of foods.
    fn refill_food(&mut self) {
        while self.foods.len() < self.rules.foods {
            match self.spawn_food() {
                Some(p) => self.foods.push(p),
                None => break,
            }
        }
    }

    /// A uniformly random free cell (no snake, rock or food), or `None` when there is none.
    fn spawn_food(&self) -> Option<Point> {
        let board = self.board();
        let mut occupied = self.walls.clone();
        for &p in self.snake.iter().chain(&self.foods) {
            if board.contains(p) {
                occupied[board.idx(p)] = true;
            }
        }
        let free = occupied.iter().filter(|o| !**o).count();
        if free == 0 {
            return None;
        }
        let mut pick = rng().random_range(0..free);
        for x in 0..board.w {
            for y in 0..board.h {
                let p = Point { x, y };
                if !occupied[board.idx(p)] {
                    if pick == 0 {
                        return Some(p);
                    }
                    pick -= 1;
                }
            }
        }
        None
    }

    pub fn next_head(&self, dir: Dir) -> Point {
        self.head().step(dir)
    }

    pub fn is_fatal(&self, dir: Dir) -> bool {
        fatal_at(
            self.board(),
            &self.walls,
            self.snake.len(),
            self.snake.iter().copied(),
            &self.foods,
            self.next_head(dir),
        )
    }

    pub fn step(&mut self, dir: Dir) {
        if self.game_over {
            return;
        }
        if self.is_fatal(dir) {
            self.game_over = true;
            self.direction = dir;
            return;
        }
        let p = self.next_head(dir);
        let eaten = self.foods.iter().position(|&f| f == p);
        self.snake.push_front(p);
        if let Some(i) = eaten {
            self.score += 1;
            self.foods.swap_remove(i);
            self.refill_food();
        } else {
            self.snake.pop_back();
        }
        self.direction = dir;
    }

    /// A game in exactly the position a snapshot describes.
    pub fn from_snapshot(snap: &GameSnapshot) -> Self {
        Self {
            width: snap.width,
            height: snap.height,
            snake: snap.snake.iter().copied().collect(),
            direction: Dir::Right,
            score: 0,
            game_over: false,
            foods: snap.foods.clone(),
            obstacles: snap.obstacles.clone(),
            rules: Rules {
                foods: snap.foods.len().max(1),
                obstacles: !snap.obstacles.is_empty(),
            },
            walls: snap.walls.clone(),
        }
    }

    /// Share of the open (non-rock) cells the snake covers.
    pub fn board_fill_ratio(&self) -> f64 {
        let open = self.width * self.height - self.obstacles.len();
        self.snake.len() as f64 / open.max(1) as f64
    }
}

/// An owned copy of what the inference thread needs to see.
#[derive(Clone, Debug)]
pub struct GameSnapshot {
    pub width: usize,
    pub height: usize,
    pub snake: Vec<Point>,
    pub foods: Vec<Point>,
    pub obstacles: Vec<Point>,
    walls: Vec<bool>,
}

impl GameSnapshot {
    pub fn from_game(game: &SnakeGame) -> Self {
        Self {
            width: game.width,
            height: game.height,
            snake: game.snake.iter().copied().collect(),
            foods: game.foods.clone(),
            obstacles: game.obstacles.clone(),
            walls: game.walls.clone(),
        }
    }

    pub fn head(&self) -> Point {
        self.snake[0]
    }

    pub fn is_fatal(&self, dir: Dir) -> bool {
        fatal_at(
            Board::new(self.width, self.height),
            &self.walls,
            self.snake.len(),
            self.snake.iter().copied(),
            &self.foods,
            self.head().step(dir),
        )
    }

    /// Path distance from every cell to its nearest food, walking around rocks and the body.
    fn food_distances(&self) -> Vec<Option<u32>> {
        let board = Board::new(self.width, self.height);
        let mut blocked = self.walls.clone();
        for &p in self.snake.iter().take(self.snake.len().saturating_sub(1)) {
            if board.contains(p) {
                blocked[board.idx(p)] = true;
            }
        }
        let mut dist = vec![None; board.cells()];
        let mut queue = VecDeque::new();
        for &f in &self.foods {
            if board.contains(f) && dist[board.idx(f)].is_none() {
                dist[board.idx(f)] = Some(0);
                queue.push_back(f);
            }
        }
        while let Some(p) = queue.pop_front() {
            let d = dist[board.idx(p)].unwrap_or(0);
            for dir in Dir::ALL {
                let n = p.step(dir);
                if board.contains(n) && !blocked[board.idx(n)] && dist[board.idx(n)].is_none() {
                    dist[board.idx(n)] = Some(d + 1);
                    queue.push_back(n);
                }
            }
        }
        dist
    }

    /// The food the snake would reach first, by path length (Manhattan when none is reachable).
    pub fn nearest_food(&self) -> Option<Point> {
        let board = Board::new(self.width, self.height);
        let h = self.head();
        let dist = self.food_distances();
        let mut blocked = self.walls.clone();
        for &p in self.snake.iter().take(self.snake.len().saturating_sub(1)) {
            if board.contains(p) {
                blocked[board.idx(p)] = true;
            }
        }
        let mut seen = vec![false; board.cells()];
        let mut queue = VecDeque::from([h]);
        if board.contains(h) {
            seen[board.idx(h)] = true;
        }
        while let Some(p) = queue.pop_front() {
            if p != h && self.foods.contains(&p) {
                return Some(p);
            }
            for dir in Dir::ALL {
                let n = p.step(dir);
                if board.contains(n)
                    && !blocked[board.idx(n)]
                    && !seen[board.idx(n)]
                    && dist[board.idx(n)].is_some()
                {
                    seen[board.idx(n)] = true;
                    queue.push_back(n);
                }
            }
        }
        self.foods
            .iter()
            .min_by_key(|f| (f.x - h.x).abs() + (f.y - h.y).abs())
            .copied()
    }

    /// For each direction (in `Dir::ALL` order): does it shorten the path to the nearest food?
    /// Falls back to Manhattan distance when no food can be reached.
    pub fn toward_food(&self) -> [bool; 4] {
        let board = Board::new(self.width, self.height);
        let h = self.head();
        let dist = self.food_distances();
        let at = |p: Point| {
            if board.contains(p) {
                dist[board.idx(p)]
            } else {
                None
            }
        };
        let best = Dir::ALL.iter().filter_map(|d| at(h.step(*d))).min();
        let mut out = [false; 4];
        match best {
            Some(m) => {
                for d in Dir::ALL {
                    out[d.index()] = at(h.step(d)) == Some(m);
                }
            }
            None => {
                if let Some(f) = self.foods.iter().min_by_key(|f| (f.x - h.x).abs() + (f.y - h.y).abs()) {
                    let md = |p: Point| (f.x - p.x).abs() + (f.y - p.y).abs();
                    for d in Dir::ALL {
                        out[d.index()] = md(h.step(d)) < md(h);
                    }
                }
            }
        }
        out
    }
}

/// Reachable cell count from `start`, walking only through unblocked in-bounds cells. `start` itself
/// is counted even if it lies off the board (a fatal move), matching the original behaviour.
fn flood_fill_area(
    board: Board,
    start: Point,
    blocked: &[bool],
    seen: &mut [bool],
    queue: &mut Vec<Point>,
) -> usize {
    if board.contains(start) && blocked[board.idx(start)] {
        return 0;
    }
    seen.fill(false);
    queue.clear();
    if board.contains(start) {
        seen[board.idx(start)] = true;
    }
    queue.push(start);
    let mut count = 1;
    let mut head = 0;
    while head < queue.len() {
        let p = queue[head];
        head += 1;
        for dir in Dir::ALL {
            let n = p.step(dir);
            if board.contains(n) {
                let i = board.idx(n);
                if !blocked[i] && !seen[i] {
                    seen[i] = true;
                    count += 1;
                    queue.push(n);
                }
            }
        }
    }
    count
}

/// Is any cell in `goals` reachable from `start` through unblocked cells?
fn bfs_any_goal(
    board: Board,
    start: Point,
    goals: &[Point],
    blocked: &[bool],
    seen: &mut [bool],
    queue: &mut Vec<Point>,
) -> bool {
    if goals.contains(&start) {
        return true;
    }
    seen.fill(false);
    queue.clear();
    if board.contains(start) {
        seen[board.idx(start)] = true;
    }
    queue.push(start);
    let mut head = 0;
    while head < queue.len() {
        let p = queue[head];
        head += 1;
        for dir in Dir::ALL {
            let n = p.step(dir);
            if goals.contains(&n) {
                return true;
            }
            if board.contains(n) {
                let i = board.idx(n);
                if !blocked[i] && !seen[i] {
                    seen[i] = true;
                    queue.push(n);
                }
            }
        }
    }
    false
}

/// `(dead-end risk, food reachable)` after executing `dir`: risk is `1 - reachable area / snake
/// length`, clamped to `[0, 1]`; reachability is 1.0 when a path to any food exists.
pub fn estimate_risk_and_reachability(game: &SnakeGame, dir: Dir) -> (f64, f64) {
    let board = game.board();
    let new_head = game.next_head(dir);
    let ate = game.foods.contains(&new_head);

    // Body after the move, minus its head: the old body, without the tail unless we ate.
    let keep = if ate {
        game.snake.len()
    } else {
        game.snake.len().saturating_sub(1)
    };
    let body_len = keep + 1;
    let mut blocked = game.walls.clone();
    for &p in game.snake.iter().take(keep) {
        if board.contains(p) {
            blocked[board.idx(p)] = true;
        }
    }

    let mut seen = vec![false; board.cells()];
    let mut queue = Vec::with_capacity(board.cells());
    let area = flood_fill_area(board, new_head, &blocked, &mut seen, &mut queue);
    let risk = (1.0 - area as f64 / body_len.max(1) as f64).clamp(0.0, 1.0);
    let reachable = !game.foods.is_empty()
        && bfs_any_goal(board, new_head, &game.foods, &blocked, &mut seen, &mut queue);
    (risk, if reachable { 1.0 } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The original `HashSet`-based implementation, kept as the oracle.
    mod reference {
        use super::super::{Dir, Point, SnakeGame};
        use std::collections::{HashSet, VecDeque};

        pub fn flood_fill_area(
            start: Point,
            blocked: &HashSet<Point>,
            width: usize,
            height: usize,
        ) -> usize {
            if blocked.contains(&start) {
                return 0;
            }
            let mut seen = HashSet::from([start]);
            let mut q = VecDeque::from([start]);
            while let Some(p) = q.pop_front() {
                for d in Dir::ALL {
                    let (dx, dy) = d.delta();
                    let n = Point {
                        x: p.x + dx,
                        y: p.y + dy,
                    };
                    if n.x >= 0
                        && n.x < width as i32
                        && n.y >= 0
                        && n.y < height as i32
                        && !blocked.contains(&n)
                        && seen.insert(n)
                    {
                        q.push_back(n);
                    }
                }
            }
            seen.len()
        }

        pub fn bfs_path_exists(
            start: Point,
            goal: Point,
            blocked: &HashSet<Point>,
            width: usize,
            height: usize,
        ) -> bool {
            if start == goal {
                return true;
            }
            let mut seen = HashSet::from([start]);
            let mut q = VecDeque::from([start]);
            while let Some(p) = q.pop_front() {
                for d in Dir::ALL {
                    let (dx, dy) = d.delta();
                    let n = Point {
                        x: p.x + dx,
                        y: p.y + dy,
                    };
                    if n == goal {
                        return true;
                    }
                    if n.x >= 0
                        && n.x < width as i32
                        && n.y >= 0
                        && n.y < height as i32
                        && !blocked.contains(&n)
                        && seen.insert(n)
                    {
                        q.push_back(n);
                    }
                }
            }
            false
        }

        pub fn estimate(game: &SnakeGame, dir: Dir) -> (f64, f64) {
            let new_head = game.next_head(dir);
            let ate = game.foods.contains(&new_head);
            let mut body_after = game.snake.clone();
            body_after.push_front(new_head);
            if !ate {
                body_after.pop_back();
            }
            let blocked: HashSet<Point> = body_after
                .iter()
                .skip(1)
                .chain(&game.obstacles)
                .copied()
                .collect();
            let area = flood_fill_area(new_head, &blocked, game.width, game.height);
            let risk = (1.0 - area as f64 / body_after.len().max(1) as f64).clamp(0.0, 1.0);
            let reachable = game
                .foods
                .iter()
                .any(|&f| bfs_path_exists(new_head, f, &blocked, game.width, game.height));
            (risk, if reachable { 1.0 } else { 0.0 })
        }

        pub fn is_fatal(game: &SnakeGame, dir: Dir) -> bool {
            let p = game.next_head(dir);
            if p.x < 0 || p.x >= game.width as i32 || p.y < 0 || p.y >= game.height as i32 {
                return true;
            }
            if game.obstacles.contains(&p) {
                return true;
            }
            let ate = game.foods.contains(&p);
            let end = if ate {
                game.snake.len()
            } else {
                game.snake.len().saturating_sub(1)
            };
            game.snake.iter().take(end).any(|&x| x == p)
        }
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % n
        }
    }

    /// A random legal position: the snake is a self-avoiding random walk, rocks and foods on free cells.
    fn random_game(r: &mut Lcg) -> SnakeGame {
        let (w, h) = (10 + r.next(20) as usize, 8 + r.next(12) as usize);
        let mut g = SnakeGame::with_rules(
            w,
            h,
            Rules {
                foods: 1,
                obstacles: false,
            },
        );
        let target = 3 + r.next((w * h / 2) as u64) as usize;
        let mut cells = vec![Point {
            x: r.next(w as u64) as i32,
            y: r.next(h as u64) as i32,
        }];
        let mut guard = 0;
        while cells.len() < target && guard < 5000 {
            guard += 1;
            let d = Dir::ALL[r.next(4) as usize];
            let n = cells[cells.len() - 1].step(d);
            if n.x >= 0 && n.x < w as i32 && n.y >= 0 && n.y < h as i32 && !cells.contains(&n) {
                cells.push(n);
            }
        }
        g.snake = cells.into_iter().rev().collect();
        let occupied: HashSet<Point> = g.snake.iter().copied().collect();
        let mut free: Vec<Point> = (0..w as i32)
            .flat_map(|x| (0..h as i32).map(move |y| Point { x, y }))
            .filter(|p| !occupied.contains(p))
            .collect();
        let (n_rocks, n_foods) = (r.next(12), 1 + r.next(4));
        let mut take = |n: u64| {
            let mut out = Vec::new();
            for _ in 0..n {
                if free.is_empty() {
                    break;
                }
                out.push(free.swap_remove(r.next(free.len() as u64) as usize));
            }
            out
        };
        let rocks = take(n_rocks);
        let foods = take(n_foods);
        g.set_obstacles(rocks);
        g.foods = foods;
        g.rules.foods = g.foods.len().max(1);
        g
    }

    #[test]
    fn risk_reachability_and_fatality_match_reference() {
        let mut r = Lcg(1234);
        for _ in 0..3000 {
            let g = random_game(&mut r);
            let snap = GameSnapshot::from_game(&g);
            for d in Dir::ALL {
                assert_eq!(
                    estimate_risk_and_reachability(&g, d),
                    reference::estimate(&g, d),
                    "{g:?} {d:?}"
                );
                assert_eq!(g.is_fatal(d), reference::is_fatal(&g, d));
                assert_eq!(snap.is_fatal(d), reference::is_fatal(&g, d));
            }
        }
    }

    #[test]
    fn off_board_start_matches_reference() {
        // Executing a fatal move (shield off) starts the flood fill outside the board.
        let mut g = SnakeGame::new(12, 9);
        g.snake = VecDeque::from([
            Point { x: 0, y: 4 },
            Point { x: 1, y: 4 },
            Point { x: 2, y: 4 },
        ]);
        g.foods = vec![Point { x: 8, y: 2 }];
        g.set_obstacles(vec![Point { x: 0, y: 3 }]);
        for d in Dir::ALL {
            assert_eq!(
                estimate_risk_and_reachability(&g, d),
                reference::estimate(&g, d)
            );
        }
    }

    #[test]
    fn step_matches_reference_rules() {
        let mut r = Lcg(99);
        for _ in 0..300 {
            let mut g = random_game(&mut r);
            for _ in 0..40 {
                if g.game_over {
                    break;
                }
                let d = Dir::ALL[r.next(4) as usize];
                let before = g.clone();
                let fatal = reference::is_fatal(&before, d);
                g.step(d);
                assert_eq!(g.game_over, fatal);
                assert_eq!(g.direction, d);
                if fatal {
                    assert_eq!(g.snake, before.snake);
                } else {
                    let ate = before.foods.contains(&before.next_head(d));
                    assert_eq!(g.score, before.score + ate as u32);
                    assert_eq!(g.head(), before.next_head(d));
                    assert_eq!(g.snake.len(), before.snake.len() + ate as usize);
                    assert!(!g.foods.contains(&g.head()), "eaten food left on the board");
                    for f in &g.foods {
                        assert!(!g.snake.contains(f), "food spawned on the snake");
                        assert!(!g.obstacles.contains(f), "food spawned on a rock");
                    }
                }
            }
        }
    }

    #[test]
    fn food_is_uniform_over_free_cells_and_none_when_full() {
        let one = Rules {
            foods: 1,
            obstacles: false,
        };
        let mut g = SnakeGame::with_rules(4, 3, one);
        g.foods.clear();
        g.snake = (0..4)
            .flat_map(|x| (0..3).map(move |y| Point { x, y }))
            .filter(|p| !(p.x == 3 && p.y == 2))
            .collect();
        assert_eq!(g.spawn_food(), Some(Point { x: 3, y: 2 }));
        g.snake.push_back(Point { x: 3, y: 2 });
        assert_eq!(g.spawn_food(), None);

        let mut g = SnakeGame::with_rules(6, 4, one);
        g.foods.clear();
        let mut counts = std::collections::HashMap::new();
        for _ in 0..12_000 {
            *counts.entry(g.spawn_food().unwrap()).or_insert(0usize) += 1;
        }
        assert_eq!(
            counts.len(),
            6 * 4 - 3,
            "every free cell should be reachable"
        );
        assert!(counts.values().all(|&c| c > 300 && c < 800), "{counts:?}");
    }

    #[test]
    fn obstacles_keep_the_board_connected_and_the_spawn_clear() {
        for _ in 0..200 {
            let g = SnakeGame::new(32, 20);
            assert!(!g.obstacles.is_empty());
            let board = g.board();
            let mut seen = vec![false; board.cells()];
            let mut queue = Vec::new();
            let open = board.cells() - g.obstacles.len();
            assert_eq!(
                flood_fill_area(board, g.head(), &g.walls, &mut seen, &mut queue),
                open
            );
            for p in &g.snake {
                assert!(!g.is_obstacle(*p));
            }
            assert!(!g.is_fatal(Dir::Right));
            assert_eq!(g.foods.len(), g.rules.foods);
        }
    }

    #[test]
    fn eating_one_of_several_foods_keeps_the_count() {
        let mut g = SnakeGame::with_rules(
            20,
            12,
            Rules {
                foods: 4,
                obstacles: false,
            },
        );
        let target = g.next_head(Dir::Right);
        g.foods[0] = target;
        g.step(Dir::Right);
        assert_eq!(g.score, 1);
        assert_eq!(g.foods.len(), 4);
        assert!(!g.foods.contains(&g.head()));
    }

    #[test]
    fn toward_food_walks_around_a_rock() {
        let mut g = SnakeGame::with_rules(
            12,
            9,
            Rules {
                foods: 1,
                obstacles: false,
            },
        );
        g.snake = VecDeque::from([
            Point { x: 4, y: 4 },
            Point { x: 3, y: 4 },
            Point { x: 2, y: 4 },
        ]);
        g.foods = vec![Point { x: 7, y: 4 }];
        g.set_obstacles((2..7).map(|y| Point { x: 5, y }).collect());
        let snap = GameSnapshot::from_game(&g);
        let t = snap.toward_food();
        assert!(!t[Dir::Right.index()], "right runs into the rock wall");
        assert!(t[Dir::Up.index()] || t[Dir::Down.index()]);
        assert_eq!(snap.nearest_food(), Some(Point { x: 7, y: 4 }));
    }
}
