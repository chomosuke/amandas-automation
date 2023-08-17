#![allow(
    unused_imports,
    dead_code,
    clippy::needless_range_loop,
    unused_labels,
    clippy::comparison_chain,
    clippy::type_complexity
)]
use std::{
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    fs,
    io::{stdin, stdout, BufReader},
    iter,
    mem::{self, swap},
    ops::Deref,
    path::Path,
    sync::{Arc, Mutex},
    thread, usize,
};

use rand::{distributions::Bernoulli, random, seq::SliceRandom, thread_rng, Rng};

fn simulate(mut grid: Vec<Vec<u8>>) -> (usize, usize) {
    // let mut pos = vec![vec![false; D]; D];
    let mut i = D as i64 / 2;
    let mut j = D as i64 / 2;
    let mut dir = 1;
    let mut count = 0;
    let mut center = 0;
    while i >= 0 && j >= 0 && i < D as i64 && j < D as i64 {
        // pos[i as usize][j as usize] = true;

        let cur = grid[i as usize][j as usize];
        if cur != 0 {
            dir = cur;
        }

        if cur != 0 {
            grid[i as usize][j as usize] = if cur <= 4 { cur + 4 } else { cur - 4 };
        }

        match dir {
            1 => {
                j += 1;
            }
            2 => {
                j += 1;
                i += 1;
            }
            3 => {
                i += 1;
            }
            4 => {
                j -= 1;
                i += 1;
            }
            5 => {
                j -= 1;
            }
            6 => {
                j -= 1;
                i -= 1;
            }
            7 => {
                i -= 1;
            }
            8 => {
                j += 1;
                i -= 1;
            }
            _ => panic!(),
        }

        count += 1;
        // center += 10 - (i - D as i64 / 2).abs() - (j - D as i64 / 2).abs();
        if (i - D as i64 / 2).abs() + (j - D as i64 / 2).abs() < 1 {
            center += 1;
        }
    }
    (count, center as usize)
}

const D: usize = 21;

fn to_string(grid: &((usize, usize), Vec<Vec<u8>>)) -> String {
    let mut str = format!("{} {}", grid.0 .0, grid.0 .1).into_bytes();
    for row in &grid.1 {
        str.push(b'\n');
        for cell in row {
            str.push(cell + b'0');
        }
    }
    str.push(b'\n');
    String::from_utf8(str).unwrap()
}

fn main() {
    let children_size = 1;
    let probability = 0.001;
    let dist = Binomial::new(D as u64 * D as u64, probability).unwrap();

    let saved = "saved.txt";
    let mut rng = thread_rng();
    if !Path::new(saved).exists() {
        let grids = Arc::new(
            (0..100)
                .map(|_| {
                    let g = (0..D)
                        .map(|i| {
                            (0..D)
                                .map(|j| {
                                    let j = if j > D / 2 {
                                        -rng.gen_range(0..=1)
                                    } else if j < D / 2 {
                                        rng.gen_range(0..=1)
                                    } else {
                                        0
                                    };
                                    let i = if i > D / 2 {
                                        -rng.gen_range(0..=1)
                                    } else if i < D / 2 {
                                        rng.gen_range(0..=1)
                                    } else {
                                        0
                                    };

                                    match (i, j) {
                                        (0, 0) => 0,
                                        (0, 1) => 1,
                                        (1, 1) => 2,
                                        (1, 0) => 3,
                                        (1, -1) => 4,
                                        (0, -1) => 5,
                                        (-1, -1) => 6,
                                        (-1, 0) => 7,
                                        (-1, 1) => 8,
                                        _ => panic!(),
                                    }
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    (simulate(g.clone()), g)
                })
                .collect::<Vec<_>>(),
        );
        fs::write(saved, grids.iter().map(to_string).collect::<String>()).unwrap();
    }
    let grids = fs::read_to_string(saved).unwrap();
    let mut grids = grids.split('\n').collect::<Vec<_>>();
    grids.pop();
    let grids: Vec<((usize, usize), Vec<Vec<u8>>)> = grids
        .chunks(22)
        .map(|ls| {
            let mut s = ls[0].split(' ');
            let n = s.next().unwrap().parse().unwrap();
            let c = s.next().unwrap().parse().unwrap();
            let grid = ls[1..22]
                .iter()
                .map(|l| l.as_bytes().iter().map(|&c| c - b'0').collect())
                .collect();
            ((n, c), grid)
        })
        .collect();
    let mut grids = Arc::new(grids);

    let mut count = 0;
    loop {
        let gen_size = grids[0].0 .0 / 16;
        let gen_size = gen_size.max(100);

        // generate the children
        let num_thread = 8;
        let handles = (0..num_thread)
            .map(|i| {
                let grids = Arc::clone(&grids);
                thread::spawn(move || {
                    let mut rng = thread_rng();
                    let mut children = Vec::with_capacity(children_size * grids.len() / num_thread);
                    for (_, grid) in &grids[i * (grids.len() / num_thread)
                        ..if i == num_thread - 1 {
                            grids.len()
                        } else {
                            (i + 1) * (grids.len() / num_thread)
                        }]
                    {
                        for _ in 0..children_size {
                            let mut child = grid.clone();
                            let n = 1 + dist.sample(&mut rng);
                            for _ in 0..n {
                                let i = rng.gen_range(0..D);
                                let j = rng.gen_range(0..D);
                                let c = rng.gen_range(1..=8);
                                child[i][j] += c;
                                child[i][j] %= 9;
                            }
                            children.push((simulate(child.clone()), child));
                        }
                    }
                    children
                })
            })
            .collect::<Vec<_>>();

        let mut children = Vec::clone(&grids);
        for h in handles {
            children.append(&mut h.join().unwrap());
        }

        children.sort_unstable_by_key(|&((k, _c), _)| {
            (
                usize::MAX - k, //
                usize::MAX - _c,
            )
        });
        children.dedup_by_key(|&mut ((k, _), _)| k);
        children.truncate(gen_size);
        let most = children[0].0 .0;
        children.retain(|&((k, _), _)| most - k < 100 || rng.gen_range(0..3) != 0);//rng.gen_range(0..most) < k);

        grids = children.into();

        println!("start");
        print!("{}", to_string(&grids[0]));
        print!("{}", to_string(&grids[grids.len() - 1]));

        println!("{count} {}", grids.len());

        count += 1;
        if count % 20 == 0 {
            fs::write(saved, grids.iter().map(to_string).collect::<String>()).unwrap();
        }
    }
}

mod io {
    use std::collections::{HashSet, VecDeque};
    use std::fmt::Display;
    use std::io::{BufReader, BufWriter, Lines, Read, Write};
    use std::marker::PhantomData;
    use std::{any::type_name, io::BufRead, str::FromStr};

    pub struct ScannerIter<'a, R: Read, T> {
        remaining: usize,
        sc: &'a mut Scanner<R>,
        item: PhantomData<T>,
    }

    impl<R: Read, T: FromStr> Iterator for ScannerIter<'_, R, T> {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.remaining == 0 {
                None
            } else {
                self.remaining -= 1;
                Some(self.sc.next::<T>())
            }
        }
    }

    pub struct Scanner<R: Read> {
        tokens: VecDeque<String>,
        delimiters: Option<HashSet<char>>,
        lines: Lines<BufReader<R>>,
    }
    impl<R: Read> Scanner<R> {
        pub fn new(source: R) -> Self {
            Self {
                tokens: VecDeque::new(),
                delimiters: None,
                lines: BufReader::new(source).lines(),
            }
        }

        pub fn with_delimiters(source: R, delimiters: &[char]) -> Self {
            Self {
                tokens: VecDeque::new(),
                delimiters: Some(delimiters.iter().copied().collect()),
                lines: BufReader::new(source).lines(),
            }
        }

        pub fn next<T: FromStr>(&mut self) -> T {
            let token = loop {
                let front = self.tokens.pop_front();
                if let Some(token) = front {
                    break token;
                }
                self.receive_input();
            };
            token
                .parse::<T>()
                .unwrap_or_else(|_| panic!("input {} isn't a {}", token, type_name::<T>()))
        }

        pub fn next_n<T: FromStr>(&mut self, n: usize) -> ScannerIter<'_, R, T> {
            ScannerIter {
                remaining: n,
                sc: self,
                item: PhantomData,
            }
        }

        pub fn next_line(&mut self) -> String {
            assert!(self.tokens.is_empty(), "You have unprocessed token");
            self.lines
                .next()
                .and_then(|e| e.ok())
                .expect("Failed to read.")
        }

        fn receive_input(&mut self) {
            let line = self
                .lines
                .next()
                .and_then(|e| e.ok())
                .expect("Failed to read.");
            if let Some(delimiters) = &self.delimiters {
                for token in line.split(|c| delimiters.contains(&c)) {
                    self.tokens.push_back(token.to_string());
                }
            } else {
                for token in line.split_whitespace() {
                    self.tokens.push_back(token.to_string());
                }
            }
        }
    }

    pub struct Printer<W: Write> {
        writer: BufWriter<W>,
    }
    impl<W: Write> Printer<W> {
        pub fn new(destination: W) -> Self {
            Self {
                writer: BufWriter::new(destination),
            }
        }

        pub fn print(&mut self, s: impl Display) {
            self.writer
                .write_all(s.to_string().as_bytes())
                .expect("print failed.");
        }

        pub fn print_bytes(&mut self, b: &[u8]) {
            self.writer.write_all(b).expect("print_bytes failed.");
        }

        pub fn println(&mut self, s: impl Display) {
            self.print(s);
            self.newline();
        }

        pub fn newline(&mut self) {
            self.print_bytes(&[b'\n']);
        }

        pub fn print_iter(&mut self, mut iter: impl Iterator<Item = impl Display>) {
            if let Some(e) = iter.next() {
                self.print(&e);
                for e in iter {
                    self.print_bytes(&[b' ']);
                    self.print(&e);
                }
            }
            self.newline();
        }
    }
    impl<W: Write> Drop for Printer<W> {
        fn drop(&mut self) {
            self.writer
                .flush()
                .expect("flush failed when dropping Printer.");
        }
    }
}
#[allow(unused_imports)]
use io::*;
use rand_distr::{Binomial, Distribution};
