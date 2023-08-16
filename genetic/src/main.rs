#![allow(unused_imports, dead_code, clippy::needless_range_loop, unused_labels)]
use std::{
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    fs,
    io::{stdin, stdout, BufReader},
    iter,
    mem::{self, swap},
    ops::Deref,
    sync::{Arc, Mutex},
    thread, usize,
};

use rand::{distributions::Bernoulli, random, thread_rng, Rng};

fn simulate(mut grid: Vec<Vec<u8>>) -> usize {
    // let mut pos = vec![vec![false; 21]; 21];
    let mut i = 10i64;
    let mut j = 10i64;
    let mut dir = 1;
    let mut count = 0;
    while i >= 0 && j >= 0 && i < 21 && j < 21 {
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
    }
    count
}

fn main() {
    let mut sc = Scanner::new(stdin());
    let gen_size = sc.next::<usize>();
    let children_size = sc.next::<usize>();
    let probability = sc.next::<f64>();
    let dist = Binomial::new(21 * 21, probability).unwrap();
    let mut rng = thread_rng();
    let mut grids = Arc::new(
        (0..gen_size)
            .map(|_| {
                let g = (0..21)
                    .map(|_| (0..21).map(|_| rng.gen_range(0..=8u8)).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                (simulate(g.clone()), g)
            })
            .collect::<Vec<_>>(),
    );

    loop {
        // generate the children
        let num_thread = 16;
        let handles = (0..num_thread)
            .map(|i| {
                let grids = Arc::clone(&grids);
                thread::spawn(move || {
                    let mut rng = thread_rng();
                    let mut children = Vec::with_capacity(children_size * grids.len() / num_thread);
                    for (_, grid) in
                        &grids[i * (grids.len() / num_thread)..(i + 1) * (grids.len() / num_thread)]
                    {
                        for _ in 0..children_size {
                            let mut child = grid.clone();
                            let n = dist.sample(&mut rng);
                            for _ in 0..n {
                                let i = rng.gen_range(0..21);
                                let j = rng.gen_range(0..21);
                                child[i][j] = rng.gen_range(0..=8);
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

        children.sort_unstable_by_key(|&(k, _)| usize::MAX - k);
        // children.dedup_by_key(|&mut (k, _)| k);
        children.dedup();
        children.truncate(gen_size);

        grids = children.into();

        println!("{}", (grids[0].0));
        for row in &(grids[0].1) {
            for cell in row {
                print!("{cell}");
            }
            println!();
        }
        println!("{}", (grids[grids.len() - 1].0));
        for row in &(grids[grids.len() - 1].1) {
            for cell in row {
                print!("{cell}");
            }
            println!();
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
