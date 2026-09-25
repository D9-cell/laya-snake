#![recursion_limit = "256"]
mod ai;
mod app;
mod bench;
#[cfg(target_arch = "x86_64")]
mod engine;
mod game;
mod mock;
mod scores;
mod web;

use ai::{download_checkpoint_if_needed, Decision, LayaBrain};
use anyhow::{Context, Result};
use app::BrainInfo;
use clapless_args::Args;
use game::GameSnapshot;

mod clapless_args {
    use crate::ai::EnginePref;

    #[derive(Debug, Clone)]
    pub struct Args {
        pub model: String,
        pub checkpoint: String,
        pub width: usize,
        pub height: usize,
        pub threads: Option<usize>,
        pub engine: EnginePref,
        pub bench: Option<usize>,
        pub verify: Option<usize>,
        pub web: Option<u16>,
        pub host: String,
        pub mock: bool,
    }

    impl Args {
        pub fn parse() -> Self {
            let mut args = std::env::args().skip(1).peekable();
            let mut out = Self {
                model: "convaiinnovations/laya".into(),
                checkpoint: "root".into(),
                width: 24,
                height: 14,
                threads: None,
                engine: EnginePref::Auto,
                bench: None,
                verify: None,
                web: None,
                host: "127.0.0.1".into(),
                mock: false,
            };
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--model" => out.model = args.next().unwrap_or(out.model),
                    "--checkpoint" => out.checkpoint = args.next().unwrap_or(out.checkpoint),
                    "--width" => {
                        out.width = args
                            .next()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(out.width)
                    }
                    "--height" => {
                        out.height = args
                            .next()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(out.height)
                    }
                    "--threads" => {
                        out.threads = args.next().and_then(|v| v.parse().ok()).filter(|&n| n > 0)
                    }
                    "--engine" => match args.next().as_deref() {
                        Some("candle") => out.engine = EnginePref::Candle,
                        Some("fast") | Some("auto") => out.engine = EnginePref::Auto,
                        other => eprintln!("--engine expects fast|candle, got {other:?}"),
                    },
                    "--bench" => {
                        let n = args.peek().and_then(|v| v.parse().ok());
                        if n.is_some() {
                            args.next();
                        }
                        out.bench = Some(n.unwrap_or(12));
                    }
                    "--verify" => {
                        let n = args.peek().and_then(|v| v.parse().ok());
                        if n.is_some() {
                            args.next();
                        }
                        out.verify = Some(n.unwrap_or(24));
                    }
                    "--web" => {
                        let n = args.peek().and_then(|v| v.parse().ok());
                        if n.is_some() {
                            args.next();
                        }
                        out.web = Some(n.unwrap_or(8080));
                    }
                    "--host" => out.host = args.next().unwrap_or(out.host),
                    "--mock" => out.mock = true,
                    "--help" | "-h" => {
                        println!("laya-snake");
                        println!(
                            "  --model <id>        Hugging Face model id or local model directory"
                        );
                        println!("  --checkpoint <name> root | multilingual | typed");
                        println!("  --width <n>         starting board width");
                        println!("  --height <n>        starting board height");
                        println!("  --threads <n>       CPU inference threads (default: all; env LAYA_THREADS)");
                        println!("  --engine <name>     fast (default) | candle  - the reference implementation");
                        println!("  --bench [n]         time n decisions (default 12) and exit");
                        println!("  --verify [n]        check the fast engine against candle on n positions and exit");
                        println!("  --web [port]        serve the live dashboard in a browser (default port 8080)");
                        println!(
                            "  --host <addr>       address for --web to bind (default 127.0.0.1)"
                        );
                        println!("  --mock              no model: a labelled heuristic stand-in, for UI development");
                        std::process::exit(0);
                    }
                    other => eprintln!("Ignoring unknown argument: {other}"),
                }
            }
            out
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !["root", "multilingual", "typed"].contains(&args.checkpoint.as_str()) {
        anyhow::bail!("checkpoint must be root, multilingual, or typed");
    }

    if args.mock {
        let info = mock::info();
        println!("Running WITHOUT the model: {} (--mock).", info.engine);
        return launch(&args, mock::decide, info);
    }

    let model_dir = download_checkpoint_if_needed(&args.model, &args.checkpoint)
        .context("preparing Laya checkpoint")?;

    if let Some(n) = args.bench {
        return bench::bench(&model_dir, args.engine, args.threads, n);
    }
    if let Some(n) = args.verify {
        return bench::verify(&model_dir, args.threads, n);
    }

    println!(
        "Loading Laya [{}] from {}...",
        args.checkpoint,
        model_dir.display()
    );

    let brain =
        LayaBrain::load(&model_dir, args.engine, args.threads).context("loading Laya model")?;

    println!(
        "Model loaded: {} on {}. Warming up...",
        brain.engine_str(),
        brain.device_str()
    );
    let warm_ms = brain.warm_up().context("warming up the model")?;
    println!("Ready (first decision took {warm_ms:.0} ms).");

    let info = BrainInfo {
        checkpoint: brain.checkpoint().to_string(),
        engine: brain.engine_str(),
        device: brain.device_str(),
    };
    launch(
        &args,
        move |snapshot| brain.decide(snapshot).map_err(|e| format!("{e:#}")),
        info,
    )
}

/// Start the browser dashboard (`--web`) or the terminal UI.
fn launch<F>(args: &Args, decide: F, info: BrainInfo) -> Result<()>
where
    F: Fn(&GameSnapshot) -> std::result::Result<Decision, String> + Send + 'static,
{
    if let Some(port) = args.web {
        return web::run(decide, info, args.width, args.height, &args.host, port);
    }
    println!("Starting live snake. Press Q to quit.");
    app::run(decide, info, args.width, args.height)
}
