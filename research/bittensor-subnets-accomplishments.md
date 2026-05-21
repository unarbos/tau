# Bittensor Subnets — Comprehensive Accomplishments Dossier

> Snapshot date: **21 May 2026**. Goal: for every Bittensor subnet
> (netuid `1`–`128`), capture the team/brand, what it does, the marketing
> positioning currently visible on public sites + X/Twitter, and a bullet
> list of concrete accomplishments cited from public sources.
>
> Discovery sources used for the master list: taomarketcap.com, taostats.io,
> subnetradar.com, bittensor.co.in (Bittensor India), taosubnetguide.com,
> bittensor.ai, individual subnet sites/GitHub/Substack. Inline `[n-m]`
> references cite the most direct public source per claim.

## How this report was assembled

The orchestrate skill's intended fan-out (one subplanner per ~32-netuid band
plus an aggregator) is captured in `.orchestrate/bittensor-subnets/plan.json`.
That orchestration could not be executed in this Cloud Agent VM because
`CURSOR_API_KEY` is not injected (see
`.orchestrate/bittensor-subnets/attention.log`), so this research was
performed directly with web search/fetch tooling. The plan.json is left in
place so the same decomposition can be re-run once the credential is added
in Cursor Dashboard → Cloud Agents → Secrets.

## Executive summary

- **Network size**: 128 subnet slots; ~90 active. Many high-number slots are
  community-flagged "scam-until-proven-otherwise" or recently deregistered.
- **Subnets with material revenue (>$1M ARR)**: Chutes (SN64), Lium (SN51),
  Tiger Alpha (SN107), Vanta/Taoshi (SN8), Dippy (SN11), ITSAI (SN32),
  Bitcast (SN93).
- **Frontier-AI milestones from Bittensor in 2026**:
  - **Templar / Teutonic (SN3)** — Covenant-72B, largest decentralized LLM
    training ever (1.1T tokens, 67.11 MMLU beating LLaMA-2-70B on half the
    data); 80B Teutonic-LXXX now in training with 1T-parameter
    ambition.[3-1][3-2]
  - **Ridges AI (SN62)** — ~80–81.5% SWE-Bench in 45 days at ~1/300th the
    cost of centralized labs.[62-1]
  - **Gradients (SN56)** — 100% win rate vs TogetherAI/Databricks/Google
    Cloud, +42.1% mean fine-tune uplift.[56-1]
  - **Chutes (SN64)** — first Bittensor subnet to cross $100M market cap;
    34T+ tokens lifetime; ~120B/day.[64-1][64-2]
- **Largest non-AI plays**: TaoHash (SN14) hit ~2 EH/s in week 1 with 11
  miners; InfiniteHash (SN89) operates one of the world's largest BTC
  Lightning nodes; NATIX StreetVision (SN72) has 250k+ drivers / 170M+ km
  of mapped streets.
- **DeSci wave is real**: NOVA (SN68) explored 65B chemical possibilities
  in year 1; Mainframe (SN25) folded more proteins in <1 year than
  Folding@Home since Oct 2000; SafeScan (SN76) presents at the National
  Oncology Institute of Maria Skłodowska-Curie.

## Master table

| # | Subnet | Owner / Team | Primary focus | Status | Top accomplishment |
|---|---|---|---|---|---|
| 1 | Apex | Macrocosmos | GAN-style LLM evaluation | Active | Apex 3.0 game-theoretic GANs + first P2P "Battleships" competition.[1-1] |
| 2 | Omron / DSperse | Inference Labs | zkML / verifiable inference | Active | 300M+ proofs; "world's largest decentralized ZK proving cluster"; 10× speed-up.[2-1] |
| 3 | Templar → Teutonic | Templar Labs → community | Decentralized LLM pre-training | Active (rebranded) | Covenant-72B trained over open internet; SparseLoCo 146× compression.[3-1] |
| 4 | Targon | Manifold Labs | Confidential GPU compute (TVM) | Active | NVIDIA Inception, Intel TDX whitepaper; ~1,400 H200s daily.[4-1] |
| 5 | Hone (was Open Kaito) | Latent Holdings | Hierarchical reasoning toward AGI | Active (rebranded) | Targets ARC-AGI-2 plateau; embedding miners reached SOTA-competitive.[5-1] |
| 6 | Numinous (was Infinite Games) | Numinous team | Binary-event forecasting | Active | Top miner beat Google Gemini Brier baseline at 0.1772, 71.8% dir. acc.[6-1] |
| 7 | SubVortex | Eclipse Vortex | Decentralized Subtensor RPC | Active | 14M+ queries; integrated into BTCLI 8.3 default endpoint.[7-1] |
| 8 | Vanta | Taoshi | Decentralized prop-trading | Active | $30M+ rewards pool; Glitch Financial; funded accounts to $2.5M.[8-1] |
| 9 | IOTA | Macrocosmos | Frontier-scale pre-training | Active | 700M–14B LLMs beating GPT2-large/Falcon-7B; 128× activation compression.[9-1] |
| 10 | Sturdy | Sturdy / Yuma | AI-driven DeFi yield | Active | $110M+ allocated; Morpho × Gauntlet $180M+ TVL vault.[10-1] |
| 11 | Dippy | Dippy AI | Roleplay LLM + multimodal app | Active | 8M+ organic users; 1B+ messages; rev tripled H1 2025.[11-1] |
| 12 | ComputeHorde | Backbone Labs | Validator GPU compute | Active | First subnet to deploy commit-reveal vs weight copiers; SDK shipped.[12-1] |
| 13 | Data Universe | Macrocosmos | Open social-media dataset | Active | 55B+ scraped posts; 40B-row HF dataset; Gravity no-code product.[13-1] |
| 14 | TAOHash | TAOHash | Redirected BTC hashrate | Active | ~2 EH/s in week 1 with 11 miners; expansion to Kaspa/Monero/LTC planned.[14-1] |
| 15 | ORO | ORO Agents | Shopping-agent benchmarking | Active | "World's largest agent competition"; reasoning-judge anti-gaming.[15-1] |
| 16 | BitAds | FirstTensor Labs | Pay-per-sale decentralized ads | Active | First decentralized PPS ad network; full slot saturation.[16-1] |
| 17 | 404-GEN | 404 (Ben James) | Text-to-3D generation | Active | 21.5M+ 3D models; Unity Asset Store plugin; 100k/8h throughput.[17-1] |
| 18 | Cortex.t / Corcel | Corcel | Decentralized LLM inference + synthetic data | Active | Corcel Chat/Image/Duet; wandb-archived synthetic Q/A.[18-1] |
| 19 | Nineteen | Rayon Labs | Fast image + LLM inference (DSIS) | Active | Hundreds of thousands of images/week; ~29% TAO emissions across Rayon trio.[19-1] |
| 20 | Bounty Hunter (was BitAgent) | Bounty Hunter team | Open AI competition platform | Active | Hosts SWE-Bench, Berkeley FCLB, Yale Spider 2.0 with on-chain leaderboard.[20-1] |
| 22 | Desearch | Datura | Decentralized search (X/Reddit/Arxiv) | Active | Datura Console API; 246 miners; sentiment + metadata products.[22-1] |
| 23 | NicheImage / Trishool | SocialTensor / TrishoolAI | Image-gen API + AI safety | Active | NicheTensor API Dec 2024; @MakeItaQuote 10k images/day; Petri alignment auditor.[23-1] |
| 24 | Quasar | SILX AI | Long-context / decentralized MoE | Active | First enterprise contract (Ayles Flow); 99.9% recall long context.[24-1] |
| 25 | Mainframe | Macrocosmos | Protein folding / DeSci | Active | 400k+ GROMACS jobs; more proteins than Folding@Home since 2000.[25-1] |
| 26 | Kinitro (was Storb) | Three Tau | Robotics policy training | Active (rebranded) | 83% on Metaworld MT10; drone-nav solved in 1 day; public dashboard.[26-1] |
| 27 | Compute Subnet | Neural Internet | Decentralized GPU compute (legacy) | Active | Proof-of-GPU validation; Docker-isolated resource allocation API.[27-1] |
| 28 | Foundry S&P Oracle | Foundry Digital | S&P 500 directional prediction | Active | Models open-sourced on HF; objective scoring vs Yahoo Finance.[28-1] |
| 31 | Candles | CandlesTAO | Crypto candle sentiment prediction | Active | Crowdsourced on-chain sentiment indices across timeframes.[31-1] |
| 32 | It's AI | ITSAI Technologies (Dubai) | Human-vs-AI text detection | Active | #1 on MGTD; 98.3% on RAID; Chrome ext + Dubai entity; $3B+ TAM.[32-1] |
| 33 | ReadyAI | Afterparty AI | Structured data / llms.txt for AI agents | Active | 10k+ websites structured; llms.txt MCP server; $5M Blockchange raise.[33-1] |
| 34 | BitMind | BitMind Labs | Deepfake / AI-content detection | Active | Rebuilt as GAS adversarial subnet; DFD-Arena dataset; CAMO MoE.[34-1] |
| 35 | Cartha / LogicNet | 0xMarkets / LogicNet | Perp DEX liquidity / math LLM | Active | Cartha live on Base, 500x leverage, ALPHA collateral.[35-1] |
| 36 | Autoppia | Autoppia | Decentralized web agents | Active | Infinite Web Arena benchmark; TEE-secured deploys; Python SDK.[36-1] |
| 37 | Aurelius | Aurelius Protocol | AI alignment dataset/red-teaming | Active | Mainnet Dec 2025; alignment-faking stress-tests.[37-1] |
| 38 | Distributed Training | DSTRBTD | Open distributed LLM training | Active | 1.1B model completed; FineWeb + butterfly all-reduce.[38-1] |
| 39 | Basilica | One Covenant / tplr-ai | Trustless GPU marketplace | Active | First post-overhaul partnership with SN56 Gradients; integrates with SN120.[39-1] |
| 41 | Sportstensor | Sportstensor | Sports prediction meta-models | Active | Polymarket partnership; Almanac Beta skin-in-the-game miners.[41-1] |
| 42 | Masa / Gopher | Masa Finance / Gopher Lab | Real-time scraping (TEE) | Active | Mainnet Aug 2024 with 10+ Genesis validators; ~20B data points/day.[42-1] |
| 43 | Graphite | GraphiteAI | Graph combinatorial optimization | Active | 230+ miners; +7% NP-hard solves; 100% response rate.[43-1] |
| 44 | Score | Score Technologies | Computer-vision sports analytics | Active | 280 football leagues; Start in Block 2026 winner; 145× more efficient than SAM3.[44-1] |
| 45 | Gen42 | brokespace | AI software-engineering agents | Active | Production Gen42 Chat/API/CLI; SWE-Bench leaderboard publishing.[45-1] |
| 46 | RESI | RESI Labs (Astrid PLC) | Decentralized real-estate database | Active | 7,572 ZIP coverage; Plaid Partner; off-market valuation network announced.[46-1] |
| 47 | Reboot | Reboot Org | Robotics + synthetic data competitions | Active | MIT-licensed multi-modal sensor + safety stack.[47-1] |
| 48 | Quantum Compute | qBitTensor Labs | Quantum computing marketplace | Active | Quantum Rings sim; HSTAB challenge solved; paid private execution tier.[48-1] |
| 49 | Polaris Cloud | PolarisCloud AI | Raw GPU/CPU compute | Active | PoW gating; multi-tier bonus structure; polariscloud.ai live.[49-1] |
| 50 | Synth | Mode Network | Probabilistic BTC/ETH/SOL/Gold forecasts | Active | 200+ models; 1,000 paths/asset; CRPS scoring; equity expansion Jan 2026.[50-1] |
| 51 | Lium | Lium / Celium | "Airbnb for GPUs" | Active | ~$600/hr rental rev; 500 H100s in month 1; rank #8 emissions; ~90% < AWS.[51-1] |
| 52 | Dojo | Tensorplex Labs | Crowdsourced HFB + synthetic data | Active | Dojo Synthetic API w/ ~300M personas; Dojo V2 competitive GAN.[52-1] |
| 53 | Efficient Frontier | SignalPlus | Live crypto strategy discovery | Active | First Bittensor real-market trading subnet; exchange-verified PnL.[53-1] |
| 54 | MIID | Yanez Compliance | Synthetic identities for KYC/AML | Active | $900k oversubscribed seed; UAV framework live.[54-1] |
| 56 | Gradients | Rayon Labs | Zero-click fine-tune + DPO/GRPO | Active | 100% win rate vs TogetherAI/Databricks/Google; +42.1% mean uplift.[56-1] |
| 57 | Gaia | Nickel5 Inc | Global weather forecasting | Active | Microsoft Aurora base model; 5,000× faster than legacy supercomputer.[57-1] |
| 59 | Babelbit | Matthew Karas | Sub-50ms voice-to-voice translation | Active | Q1 2026 live demos; AWS Marketplace discussions Q2.[59-1] |
| 60 | Bitsec | Bitsec / ChainDefender | AI vulnerability scanning | Active | Found a Bittensor-core exploit in 10 minutes; Rust contract pagination bug.[60-1] |
| 61 | RedTeam | Innerworks | Cybersecurity competitions in prod | Active | 14 challenges run/9 deprecated; 26× improvement on AB Sniffer v1→v4.[61-1] |
| 62 | Ridges AI | Ridges AI | Best-in-world coding agent on SWE-Bench | Active | ~80–81.5% SWE-Bench in 45 days, ~1/300 cost vs centralized labs.[62-1] |
| 63 | Enigma / Quantum Innovate | qBitTensor Labs | Deep-tech competition platform | Active | Scaled 36→37 qubit sims; Bittensor Treasury Wallet pioneer.[63-1] |
| 64 | Chutes | Rayon Labs | Serverless AI compute / inference layer | Active | First subnet >$100M MC; 34T+ tokens; ~120B/day; $5.5M ARR.[64-1] |
| 65 | TPN | TAOFu Labs | Residential-IP decentralized VPN | Active | Removed 250-node cap; WhaleSurf consumer app iOS/Android; agent SOCKS5.[65-1] |
| 66 | Ninja | Project Nobi / unarbos | Coding-agent dueling tournament | Active | Project Nobi Arena head-to-head vs SN62; Claude Sonnet judge LLM.[66-1] |
| 67 | AlphaCore | AlphaCore Bittensor | Autonomous cloud DevOps | Active | Proof-of-Deployment + Firecracker microVMs; 440 TAO Bitstarter raise.[67-1] |
| 68 | NOVA | Metanova Labs | Decentralized drug discovery | Active | Y1: 65B chemical possibilities, 11M+ molecules, ONEPOT.AI 5–7-day synthesis.[68-1] |
| 70 | Vericore | dFusion AI | Large-scale semantic fact-checking | Active | 252 miners; 49 versions; semantic verification with cited evidence.[70-1] |
| 71 | Leadpoet | Leadpoet | Autonomous AI sales leads | Active | 1.1M+ validated leads; 5.67M in inventory; beta Dec 2025.[71-1] |
| 72 | StreetVision | NATIX Network | Decentralized driving imagery | Active | 250k+ drivers; 170M+ km mapped; 925M+ data points; Grab partnership.[72-1] |
| 73 | MetaHash | FX Integral / MetaHash Group | OTC ALPHA → META swap; subnet treasury | Active | First meta-mining subnet; Dutch-auction epochs; multi-domain group.[73-1] |
| 74 | Gittensor | Entrius | Reward open-source GitHub contributions | Active | Sybil-resistant PR-merge verification; semantic code scoring.[74-1] |
| 75 | Hippius | Hippius | S3-compatible decentralized storage | Active | 459+ nodes; drop-in AWS S3; quantum-safe encryption deployed.[75-1] |
| 76 | SafeScan | SAFESCAN | Skin cancer (melanoma) detection | Active | MILK10k benchmark; National Oncology Institute of Maria S-C presentations.[76-1] |
| 77 | Liquidity | Oceans Subnet team | Bittensor on-chain liquidity mining | Active | Liquidity auction + voting; Ethereum smart contracts; 244 miners.[77-1] |
| 78 | Loosh | Loosh AI (Yuma-backed) | Machine-consciousness / cognition layer | Active | Cognition Engine Beta with virtue + rights evaluators.[78-1] |
| 79 | MVTRX (was TAOS) | TAOS / MVTRX | L3 MBO exchange for dTAO + alpha | Active | Mainnet May 2025; Dynamic Incentive Structure; $65M+ 24h vol.[79-1] |
| 81 | Grail (was Patrol) | Grail team (ex-Tensora) | Decentralized RL post-training | Active | Async trainer training a 7B model; GRAIL cryptographic rollouts.[81-1] |
| 83 | CliqueAI | TopTensor | Max-clique graph optimization | Active | 192 miners + 64 validators; v0.0.13 sigmoid weight scaling.[83-1] |
| 84 | ChipForge | Tatsu Project | Decentralized chip design (RISC-V) | Active | First decentralized-designed RISC-V MCU with hardware AES/SHA; FPGA-validated.[84-1] |
| 85 | Vidaio | Vidaio (Gareth Howells) | AI video upscale + compression | Active | Upscaling live Apr 2025; dual synthetic/organic pipelines; ex-Netflix lead.[85-1] |
| 88 | Investing | Mobius Fund | Decentralized AUM / quant fund | Active | 88 Quant Fund Dec 2025; Shariah "S-Model" planned; TAO/Alpha + US stocks phases.[88-1] |
| 89 | InfiniteHash | Backend Developers Ltd | BTC mining pool + LN node | Active | Runs one of the largest BTC Lightning nodes globally.[89-1] |
| 91 | Tensorprox | Shugo LLC | Decentralized DDoS mitigation (XDP/eBPF) | Active | 14M+ pps per core; sub-microsecond filtering; multi-pillar scrubbing.[91-1] |
| 92 | Tensorclaw / StoryNet | Tensorclaw / StorynetAI | LLM-API aggregation + story-gen | Active | WebSocket router lets miners join without public IPs; throughput-based scoring.[92-1] |
| 93 | Bitcast | Bitcast | Decentralized YouTube influencer ads | Active | First flywheel subnet; 300k+ YouTube subs; Taostats campaign 32k views/<$5k.[93-1] |
| 94 | Bitsota | Alveus Labs | Decentralized AutoML | Active | AutoML-Zero-style genetic programming on CIFAR-10; quorum validation.[94-1] |
| 96 | FLock OFF | FLock.io | Crowdsourced SLM training datasets | Active | Mainnet May 2025; LoRA-on-Qwen2.5-1.5B evaluation pipeline.[96-1] |
| 97 | Distil | Arbos / unarbos | LLM knowledge distillation (Qwen 35B → ≤5.25B) | Active | Built by an AI agent on Chutes in 48 h; best students 0.24 KL div.[97-1] |
| 98 | FlameWire / ForeverMoney | Unitone Labs / ForeverMoney | RPC gateway / DeFi vault ALM | Active | 742k+ RPC reqs at 23ms; ForeverMoney $42M Halborn-audited cbBTC vault.[98-1] |
| 102 | ConnitoAI | ConnitoAI | Distributed MoE LLM training | Active | Health 69/100; 256 neurons; 6 validators.[102-1] |
| 103 | Djinn | Djinn Inc | Sports-intelligence marketplace (zk) | Active | USDC revenue via 0.5% fee; Shamir-sharded MPC validators.[103-1] |
| 106 | VoidAI | v0idai | Bittensor × Solana cross-chain liquidity | Active | wTAO+wAlpha bridge; +45% Raydium depth; +30% LP rewards.[106-1] |
| 107 | Tiger Alpha | Tiger Invests (Tao Alpha PLC) | Verifiable crypto market data oracle | Active | $2.56M ARR by Jun 2025; sister "Beta" subnet launched.[107-1] |
| 111 | OneOneOne | OneOneOne | UGC + authenticity scoring API | Active | 3M+ items/day; oneoneone.io API; SN13+SN64+SN22 integrations.[111-1] |
| 112 | Minotaur | Subnet112 team | DEX aggregator + intent solver | Active | Cryptographic solver accountability; multi-chain (ETH/Base).[112-1] |
| 114 | Level 114 | Level 114 | First gaming subnet (Minecraft) | Active | 99.7% uptime; 2,312 players online; Sep 2025 launch.[114-1] |
| 116 | TaoLend | XPEN Labs | Lending TAO vs ALPHA collateral | Active | EVM lending live at taolend.io; 20/80 lender/protocol split.[116-1] |
| 117 | BrainPlay | ShiftLayer LLC | Benchmark via gameplay | Active | Codenames live; 20Q + Super Mario coming; uses SN4 Targon TVM.[117-1] |
| 120 | Affine | Affine Foundation | Cross-subnet RL coordinator | Active | Top-3 emissions; uses 10+ subnets; sybil/decoy/copy/overfit-proof RL.[120-1] |
| 121 | sundae_bar | sundae_bar PLC (AIM: SBAR) | Generalist-agent benchmark + marketplace | Active | London AIM-listed Jun 2025; ~$182k Alpha minted since June.[121-1] |
| 122 | Bitrecs | Bitrecs | E-commerce recommendation engine | Active | V2 prompt-evolution; ε-Pareto winner-take-all; Shopify plugin.[122-1] |
| 124 | Swarm | swarm-subnet | Autonomous drone autopilot | Active | PyBullet headless tournament; 50/50 mission+speed reward; framework-agnostic.[124-1] |
| 128 | ByteLeap | ByteLeap AI | Decentralized GPU cloud (VFIO) | Active | 1,100–1,216 GPUs live across consumer + datacenter mix.[128-1] |

> Slots **40, 49 (verify), 55, 58, 69, 82, 86, 87, 90, 95, 99, 101, 104,
> 105, 108, 109, 110, 113, 115, 118, 119, 123, 125, 126, 127** are
> currently either unclaimed, inactive, recently deregistered, or sit in
> the community "scam-until-proven-otherwise" bucket per
> taosubnetguide.com and bittensor.co.in. Cross-check live status on
> taostats.io before publishing — Bittensor's monthly cleanups churn
> these aggressively.

---

## Per-subnet dossiers (active subnets)

### SN1 — Apex (Macrocosmos)

- **Pitch**: "Apex turns AI agents into autonomous researchers" — Bittensor
  competitions that evaluate *subjective* LLM outputs with GANs rather than
  static benchmarks.
- **X/Twitter**: `@macrocosmosai` (covers SN1, SN9, SN13, SN25); Substack
  at macrocosmosai.substack.com.
- **Accomplishments**:
  - Apex 3.0 (Aug 2025) — game-theoretic Generative Adversarial Mining
    for subjective LLM evaluation.[1-2]
  - First fully P2P competition (Battleships v1) Dec 2025 — miners
    duelling head-to-head, a Bittensor first.[1-3]
  - Outsourced inference to SN64 and web retrieval to SN13 to focus
    entirely on the GAN mechanism.[1-2]

[1-1] taodaily.io/apex-subnet-1-turning-ai-agents-into-autonomous-researchers-on-bittensor
[1-2] macrocosmosai.substack.com/p/apex-30-game-theoretic-ai-on-bittensor
[1-3] macrocosmosai.substack.com/p/versatility-and-flexibility-with

### SN2 — Omron / DSperse (Inference Labs)

- **Pitch (omron.ai, inferencelabs.com)**: "Verified AI for Bittensor" —
  Proof-of-Inference that the right model produced the right output.
- **X/Twitter**: `@inferencelabs`, `@omron_ai`.
- **Accomplishments**:
  - 300M+ verifiable inferences; 1,500+ unique miners; 10× proof-time
    speed-up.[2-1]
  - Feb 2026 rebrand as "world's fastest zkML proving cluster"; released
    DSperse + JSTprove frameworks.[2-1]
  - Permanent proof storage on Arweave via AR.IO (Mar 2025).[2-1]
  - 2026 partnerships: Score (Feb 2026), OpenLedger (Jan 2026), Cysic
    (Dec 2025).[2-1]

[2-1] inferencelabs.com/media-room

### SN3 — Templar → Teutonic

- **Pitch**: Frontier-scale LLMs trained over the open internet — no
  private interconnects, no specialized hardware, no whitelist.
- **X/Twitter**: `@tplr_ai`; Bittensor co-founder `@const_reborn` drives
  Teutonic continuation.
- **Accomplishments**:
  - **Covenant-72B** — largest decentralized LLM pre-training run in
    history; 72B params, 1.1T tokens, 70+ peers on commodity ~500/110 Mb/s
    links, 94.5% compute utilization, ~70 s sync rounds (down from
    8.3 min).[3-1]
  - **SparseLoCo** optimizer: 146× gradient compression (top-k + 2-bit
    quant + error feedback).[3-1]
  - **67.11 MMLU** beating LLaMA-2-70B's 65.63 with ~½ the training
    data.[3-2]
  - After Covenant AI's exit (Apr 2026), community rebuilt SN3 open-source
    as **Teutonic**; Teutonic-LXXX (80B) began training May 2026 with
    publicly-stated 1T-parameter ambition.[3-3][3-4]

[3-1] tao.media/templar-makes-history-with-72b-decentralized-ai-training-run
[3-2] eli5defi.substack.com/p/the-largest-ai-training-run-in-history
[3-3] tao.media/teutonic-subnet-begins-training-80b-ai-model-on-bittensor-marking-largest-decentralized-training-run-yet
[3-4] taodaily.io/teutonic-is-live-const-is-cooking-the-1-trillion-parameter-run-is-loading

### SN4 — Targon (Manifold Labs)

- **Pitch (manifold.inc)**: Confidential GPU compute — re-attested every
  72 minutes on NVIDIA Hopper/Blackwell GPUs with Intel TDX / AMD SEV-SNP,
  end-to-end encryption from disk to GPU.
- **X/Twitter**: `@manifoldlabs`.
- **Accomplishments**:
  - ~**1,400 H200s** daily — "in hyperscaler territory."[4-1]
  - **Intel × Manifold whitepaper** (Mar 2026) — first formal security
    paper from a Bittensor subnet with a major chip manufacturer.[4-2]
  - Accepted into **NVIDIA Inception** (Mar 2026).[4-3]
  - Powers Sybil decentralized search; six-figure SN11 Dippy inference
    deal.

[4-1] manifold.inc/releases/manifold-2.0
[4-2] cryptojist.com/intel-bittensor-subnet-targon-sn4
[4-3] tao.media/targon-joins-nvidia-inception-program

### SN5 — Hone (was Open Kaito) — Latent Holdings

- **Pitch**: Hierarchical AI reasoning toward AGI; targets the ARC-AGI-2
  plateau.
- **X/Twitter**: `@latent_holdings`.
- **Accomplishments**:
  - Ownership transferred from Kaito to Latent Holdings May 2025.[5-1]
  - Embedding miners reached competitive vs SOTA on text-embedding
    benchmarks (Oct 2024).[5-2]
  - Incorporates LeCun JEPA/H-JEPA + the Hierarchical Reasoning Model
    (HRM).[5-1]

[5-1] iq.wiki/wiki/hone
[5-2] medium.com/@0xai.dev/sn-5-pioneering-the-decentralized-revolution-in-text-embedding-models-7eb5a68f86c1

### SN6 — Numinous (was Infinite Games)

- **Pitch**: Decentralized forecasting protocol — AI agents predict
  binary future events, benchmarked vs human + frontier LLMs.
- **X/Twitter**: `@numinous_ai`.
- **Accomplishments**:
  - Top miner UID 128 beat Google **Gemini** at Brier 0.1772 with 71.8%
    directional accuracy across 600+ live events vs 221 other agents.[6-1]
  - Calibration held at scale, not single-shot variance.[6-1]

[6-1] taodaily.io/how-a-miner-on-numinous-bittensor-subnet-6-outperforms-gemini

### SN7 — SubVortex (Eclipse Vortex)

- **Pitch (subvortex.com)**: Low-latency public subtensor endpoints —
  one-click deploy.
- **X/Twitter**: `@subvortex`.
- **Accomplishments**:
  - **14M+** queries; 237 miners; 22 validators; 111 countries.[7-1]
  - Integrated into Bittensor 8.3 BTCLI as default public endpoint.[7-1]
  - DNS across 4 geographically diverse namespace servers; Google-DNS-
    routed weighted round-robin.[7-1]

[7-1] subvortex.com

### SN8 — Vanta (Taoshi)

- **Pitch (taoshi.io)**: Decentralized prop-trading network — on-chain
  rules; "profit when you win" instead of "profit when you lose."
- **X/Twitter**: `@taoshiio`.
- **Accomplishments**:
  - **$30M+** annualized rewards pool — "largest for trading signals in
    the world."[8-1]
  - Vanta Trading funded accounts — 100% profit split, scaling to
    $2.5M; 60-day challenge, 10% max drawdown.[8-2]
  - **15,000+ $ALPHA** burned via collateral slashing since November.[8-2]
  - **Glitch Financial** wealth product launched Oct 2024.[8-3]

[8-1] taoshi.io/newsroom/competition
[8-2] taodaily.io/how-vanta-subnet-8-is-building-the-infrastructure-for-an-agent-driven-trading-economy-on-bittensor
[8-3] taoshi.io

### SN9 — IOTA (Macrocosmos)

- **Pitch (macrocosmos.ai/sn9)**: "Bittensor's biggest pre-training
  breakthrough" — trustless swarm training that doesn't require each node
  to fit the model.
- **X/Twitter**: `@macrocosmosai`.
- **Accomplishments**:
  - Original SN9 (Aug 2024) trained 700M–14B LLMs beating GPT2-large and
    Falcon-7B.[9-1]
  - IOTA architecture: **butterfly all-reduce**, **128× activation
    compression**, **CLASP** fair-attribution scheme.[9-2]
  - arXiv paper *Incentivised Orchestrated Training Architecture*
    (arXiv:2507.17766).[9-3]

[9-1] docs.macrocosmos.ai/subnets/subnet-9-iota
[9-2] arxiv.org/html/2507.17766
[9-3] macrocosmosai.substack.com/p/iota-bittensors-biggest-pretraining

### SN10 — Sturdy

- **Pitch (sturdy.finance)**: AI-optimized DeFi yield across Morpho,
  Yearn, Pendle, Ondo.
- **X/Twitter**: `@sturdyfinance`.
- **Accomplishments**:
  - **$110M+** allocated by Nov 2024.[10-1]
  - **Morpho × Gauntlet** Aggregator vault holds **$180M+ TVL** across
    four Morpho vaults — first non-Sturdy protocol routed through SN10.[10-2]
  - First subnet incubated by **Yuma Group** (DCG subsidiary).[10-1]

[10-1] building.theatlantic.com/sturdy-november-highlights-07461aa4ac3f
[10-2] building.theatlantic.com/sturdy-teams-up-with-morpho-to-offer-ai-optimized-yields

### SN11 — Dippy

- **Pitch (dippy.studio)**: Multimodal roleplay companion app + world's
  fastest LoRA training/inference for FLUX.
- **X/Twitter**: `@dippyai`.
- **Accomplishments**:
  - **8M+ organic users** (no paid ads); **1B+** messages; revenue tripled
    H1 2025; >1 hour/day avg user time.[11-1]
  - **$40k–60k/month** revenue; six-figure SN4 Targon inference deal.[11-2]
  - Multimodal expansion to image-gen + expressive TTS.[11-1]

[11-1] taodaily.io/fun-fact-dippy-gained-over-8-million-users-without-spending-a-dime-on-ads
[11-2] podscan.fm/podcasts/revenue-search-inside-bittensor-1/episodes/subnet-session-with-akshat-from-dippy-subnet-11

### SN12 — ComputeHorde (Backbone Labs)

- **Pitch (computehorde.io)**: Trustless GPU compute for validators.
- **X/Twitter**: `@computehorde`.
- **Accomplishments**:
  - First subnet to deploy **commit-reveal vs weight copiers**, with
    "executor dancing" anti-copy mitigations.[12-1]
  - SDK shipped; Grafana dashboards, governance bots, DDoS protection,
    on-chain collateral slashing.[12-1]
  - Designed for >1,000 Bittensor subnets via UID-agnostic executor
    spawning.[12-1]

[12-1] computehorde.io

### SN13 — Data Universe (Macrocosmos)

- **Pitch**: Open-source social-media dataset network — the "data layer of
  decentralized AI."
- **X/Twitter**: `@macrocosmosai`; product at gravity.macrocosmos.ai.
- **Accomplishments**:
  - **55B+** scraped posts/comments; **>40B-row** fresh X/Reddit dataset
    on Hugging Face.[13-1][13-2]
  - **Gravity** no-code data product (3,000 free credits per signup).[13-2]
  - SN13 ↔ SN57 (Gaia) integration: social signal + weather fusion.[57-2]

[13-1] docs.macrocosmos.ai/subnets/subnet-13-data-universe
[13-2] macrocosmosai.substack.com/p/data-universe-how-to-collect-real

### SN14 — TAOHash

- **Pitch**: Redirect your BTC ASIC hashrate, earn ALPHA.
- **X/Twitter**: `@taohash`.
- **Accomplishments**:
  - **~2 EH/s** with **11 active miners** in week 1.[14-2]
  - Architected to expand to **Kaspa, Monero, Litecoin**.[14-1]
  - 244 miners / 11 validators in recent dashboard.[14-3]

[14-1] subnetalpha.ai/subnet/taohash
[14-2] theopensourcepress.com/how-this-bittensor-subnet-is-reshaping-bitcoin-mining-from-the-ground-up
[14-3] bittensormarketcap.com/subnets/14

### SN15 — ORO

- **Pitch (oroagents.com)**: "World's largest agent competition" — shopping
  agents on ShoppingBench with a reasoning judge that penalizes test-pattern
  overfitting.
- **X/Twitter**: `@oroagents`.
- **Accomplishments**:
  - Open-sourced **SR25519** middleware + session management + nonce
    replay protection.[15-2]
  - Weight to top-50% miners; bottom-50% excluded; mid-race re-eval
    exploits patched.[15-1]

[15-1] wutao.app/subnet/sn15
[15-2] oroagents.com/blog/we-fixed-one-of-bittensors-biggest-problems

### SN16 — BitAds (FirstTensor Labs)

- **Pitch (bitads.ai)**: First decentralized **pay-per-sale** ad network.
- **X/Twitter**: `@bitads_ai`.
- **Accomplishments**:
  - 64 validators / 192 miners; full slot saturation.[16-1]
  - Fingerprinting + behavioral analysis filters bot traffic; sales (15%)
    + revenue (85%) weighted scoring with refund penalties.[16-2]

[16-1] subnetradar.com/subnet/16
[16-2] bitads.ai/whitepaper

### SN17 — 404-GEN

- **Pitch (404.xyz)**: Text-to-3D generative AI — "from keyboard to world."
- **X/Twitter**: `@404gen_`.
- **Accomplishments**:
  - **21.5M+** AI-generated 3D models (~40 TB).[17-1]
  - 100k+ verified 3D models every 8 hours.[17-2]
  - **Unity Asset Store** plugin — first blockchain-based 3D generation
    tool inside Unity; Unity's engineering team redesigned AI dependencies
    for commercial viability.[17-3]
  - Public HF dataset; Blender plugin + Discord bot + API.[17-2]

[17-1] taodaily.io/404-gen-from-keyboard-to-world
[17-2] doc.404.xyz
[17-3] blockchaingamer.biz/news/38206/404-bittensor-genai-text-to-3d-model-plug-in-unity-asset-store

### SN18 — Cortex.t / Corcel

- **Pitch (cortex-t.ai, corcel.io)**: Decentralized access to major LLM
  APIs + synthetic Q/A datasets.
- **X/Twitter**: `@corcel_io`.
- **Accomplishments**:
  - Synthetic dataset archived at `wandb.ai/cortex-t/synthetic-QA`.[18-1]
  - Surface products: Corcel Chat, Image Studio, Duet.[18-1]
  - Hosts "the leading LLM on Bittensor" per docs.[18-1]

[18-1] cortex-t.ai

### SN19 — Nineteen (Rayon Labs)

- **Pitch**: Fastest free inference on Bittensor — narrow focus, raw
  throughput.
- **X/Twitter**: `@rayon_labs`, `@namoray_eth`.
- **Accomplishments**:
  - Hundreds of thousands of images/week.[19-2]
  - Powers Corcel, Tao.Bot, Make It A Quote.[19-2]
  - Part of the Rayon trifecta (SN19 + SN56 + SN64) commanding ~29% of
    daily TAO emissions.[19-1]

[19-1] oakresearch.io/en/analyses/innovations/rayon-labs-subnet-leader-bittensor-tao
[19-2] learnbittensor.org/subnets/namoray/nineteen

### SN20 — Bounty Hunter (was BitAgent)

- **Pitch**: "Global engine for solving the hardest AI problems" — hosts
  Berkeley FCLB, Princeton SWE-Bench, Yale Spider 2.0.
- **X/Twitter**: `@bountyhunter_ai`.
- **Accomplishments**:
  - Open participation (no Bittensor miner stack required); all submissions
    Apache 2.0.[20-1]
  - Transparent on-chain leaderboard, TAO payouts proportional to accuracy.[20-1]

[20-1] subnetalpha.ai/subnet/bit-agent

### SN22 — Desearch (Datura)

- **Pitch (desearch.ai, datura.ai)**: Decentralized search over X, Reddit,
  Arxiv, web — privacy-preserving alternative to Google.
- **X/Twitter**: `@daturaplatform`, `@desearch_ai`.
- **Accomplishments**:
  - Multi-source real-time metadata API on Datura Console.[22-1]
  - 246 miners, 11 validators; v0.0.188 with 20+ contributors and 7
    releases on GitHub.[22-1]
  - Sentiment-analysis layer for social posts.[22-1]

[22-1] bittensormarketcap.com/subnets/22

### SN23 — NicheImage / SocialTensor + Trishool

- **Pitches**:
  - NicheImage (socialtensor.org): image-gen API bridging Web3 and Web2.
  - Trishool (trishool.ai): tri-cameral AI safety alignment subnet
    (Architects / Adversaries / Oracles).
- **X/Twitter**: `@socialtensor`, `@MakeItaQuote`.
- **Accomplishments**:
  - **NicheTensor API** Dec 2024 — among the first Bittensor subnets with
    paying Web2 customers.[23-1]
  - `@MakeItaQuote` Twitter tool generates **10,000+ images/day**.[23-1]
  - Multi-model image gen (GoJourney, FluxSchnell, OpenGeneral, etc.).[23-2]
  - Trishool evaluates LLM behavioral traits via the **Petri** alignment
    auditing agent.[23-3]

[23-1] globenewswire.com/news-release/2024/12/05/2992633/0/en/SocialTensor-Launches-NicheTensor-API
[23-2] github.com/SocialTensor/SocialTensorSubnet
[23-3] github.com/TrishoolAI/trishool-subnet

### SN24 — Quasar (SILX AI)

- **Pitch**: "Transformers 2.0" — Continuous-Time Attention Transformer
  for million-token context.
- **X/Twitter**: `@silxai`.
- **Accomplishments**:
  - First enterprise contract signed with **Ayles Flow**; mainnet live.[24-1]
  - ~**99.9% recall** across long sequences.[24-2]
  - Decentralized Mixture-of-Experts with one-to-one expert
    communication.[24-3]

[24-1] taodaily.io/bittensor-ecosystem-highlights-february-week-3
[24-2] bitstarter.ai/subnets/quasar
[24-3] taodaily.io/sn24s-eyad-discusses-long-context-ai-decentralized-moe-open-source-bittensor

### SN25 — Mainframe (Macrocosmos)

- **Pitch**: Open-source GROMACS/OpenMM protein folding — alternative to
  AlphaFold compute economics.
- **X/Twitter**: `@macrocosmosai`.
- **Accomplishments**:
  - **400,000+** GROMACS jobs and **150,000+** OpenMM jobs.[25-1]
  - **More proteins folded in <1 year than Folding@Home has since October
    2000**.[25-2]
  - First DeSci subnet on Bittensor; first to enter academic use cases.[25-2]

[25-1] docs.macrocosmos.ai/subnets/subnet-25-mainframe
[25-2] macrocosmosai.substack.com/p/bittensor-is-best-for-desci-and-mainframe

### SN26 — Kinitro (was Storb) — Three Tau

- **Pitch (kinitro.ai)**: Decentralized robotics intelligence — miners
  compete on Metaworld, drone-nav, vision-only RL.
- **X/Twitter**: `@kinitro_ai`.
- **Accomplishments**:
  - **83%** success on Metaworld MT10 (Dec 2025).[26-1]
  - Drone-nav challenge solved within **one day** of launch.[26-1]
  - Public dashboard kinitro.ai/dashboard.[26-1]
  - Storb archived after Q2 2025 rebrand.[26-2]

[26-1] taodaily.io/kinitro-is-training-ai-to-control-robots-through-competition-on-bittensor
[26-2] github.com/threetau/storb

### SN27 — Compute Subnet (Neural Internet)

- **Pitch (neuralinternet.ai)**: Permissionless cloud compute marketplace,
  Docker-isolated, proof-of-GPU validated.
- **X/Twitter**: `@neuralinternet`.
- **Accomplishments**:
  - Concluded testing-phase emissions increase in Dec 2023; auto-update
    + Docker activation enforcement shipped.[27-1]
  - Three core synapses (Specs / Allocate / Challenge).[27-2]

[27-1] medium.com/@neuralinternet/compute-subnet-incentive-increase
[27-2] docs.neuralinternet.ai/communication-protocols

### SN28 — Foundry S&P 500 Oracle (Foundry Digital)

- **Pitch**: Bring Bittensor into "the largest market in the world — the
  global economy."
- **X/Twitter**: `@FoundryServices`.
- **Accomplishments**:
  - Mainnet launch Feb 20, 2024.[28-1]
  - All models + input data open-sourced on HF for emissions eligibility —
    objective scoring against Yahoo Finance.[28-1]

[28-1] github.com/foundryservices/snpOracle

### SN31 — Candles

- **Pitch**: Crowdsourced green/red candle prediction across hourly /
  daily / weekly timeframes.
- **X/Twitter**: `@candles_tao`.
- **Accomplishments**:
  - Active dev through Dec 2025; on-chain sentiment index pooling system
    for non-experts.[31-1]

[31-1] bittensor123.com/subnets/sn31

### SN32 — It's AI

- **Pitch (itsai.tech)**: Best AI-text detector in the world — Chrome
  extension, web app, API, X bot.
- **X/Twitter**: `@itsai_tech`.
- **Accomplishments**:
  - **#1** on MGTD benchmark (>92% ROC-AUC) beating GPTZero, Originality,
    CopyLeaks.[32-1]
  - **98.3%** on RAID benchmark (ACL 2024); 99.1% avg on GRiD/HC3/GhostBuster.[32-1]
  - 0.8% FPR on ASAP 2.0 essays; 98.7% acc / ≤0.5% FPR on Arabic.[32-1]
  - **ITSAI TECHNOLOGIES – FZCO** Dubai entity (Jan 2025) targeting
    $3B+ TAM.[32-1]
  - 22k+ site visits in Feb 2025.[32-1]

[32-1] github.com/It-s-AI/llm-detection

### SN33 — ReadyAI (Afterparty AI)

- **Pitch**: Conversation Genome Project — turn raw web/podcasts into
  AI-ready datasets.
- **X/Twitter**: `@readyai`.
- **Accomplishments**:
  - **10,000+ websites** structured; target 1M domains by year-end.[33-1]
  - **llms.txt MCP server** (Apr 2026) — semantic summaries, named
    entities, topic classifications.[33-1]
  - **$5M raise** led by Blockchange Ventures (Sep 2023).[33-2]
  - ~2.51% network emissions at Oct 2024.[33-2]

[33-1] tao.media/readyai-launches-llms-txt-mcp-server-structuring-10-000-websites-for-ai-agents
[33-2] medium.com/@bittensor_player/bittensor-subnet-33-from-innovation-to-exploitation-4aae796a5f6c

### SN34 — BitMind

- **Pitch (bitmind.ai)**: Open detection of deepfakes + AI-gen media —
  image, video, audio modalities.
- **X/Twitter**: `@bitmind_ai`.
- **Accomplishments**:
  - Rebuilt as **GAS** (Generative Adversarial Subnet) — removes API costs,
    registration wars, gives full model privacy.[34-1]
  - **DFD-Arena** open-source deepfake-detection benchmark dataset.[34-2]
  - **CAMO** content-aware MoE detector.[34-2]
  - Fastest-growing subnet by emission share Aug→Sep 2024 (1% → 1.65%).[34-2]

[34-1] bitmind.ai/blog/gas-subnet-34-generative-adversarial-engine
[34-2] medium.com/bitmindlabs/bitmind-subnet-34-september-2024-recap

### SN35 — Cartha / LogicNet

- **Pitches**:
  - Cartha (0xmarkets.io): liquidity backbone for the 0xMarkets perpetuals
    DEX — up to 500x leverage, USDC collateral.
  - LogicNet: math LLM with self-executing Python.
- **X/Twitter**: `@0xmarkets`, `@logicnet_subnet`.
- **Accomplishments**:
  - Cartha live on Base Mainnet with BTC/ETH/TAO/GOLD/SILVER/JPY/EUR/GBP/
    USD pairs.[35-1]
  - LogicNet: rank-based cubic rewards, similarity + correctness scoring,
    self-executing Python for math.[35-2]

[35-1] docs.0xmarkets.io/cartha
[35-2] github.com/LogicNet-Subnet/LogicNet

### SN36 — Autoppia

- **Pitch (autoppia.com)**: Decentralized marketplace for AI Workers +
  web agents.
- **X/Twitter**: `@autoppia`.
- **Accomplishments**:
  - **Infinite Web Arena (IWA)** synthetic benchmark for web agents.[36-1]
  - Mainnet Feb 6, 2025; TEE-secured deployments; Python SDK + YAML
    configs; chainable LLM/Email interfaces.[36-2]

[36-1] github.com/autoppia/autoppia_web_agents_subnet
[36-2] asymmetricjump.substack.com/p/autoppia-sn-36-the-ai-worker-ecosystem

### SN37 — Aurelius

- **Pitch (medium.com/aurelius-protocol)**: Decentralized AI alignment —
  stress-test models for "alignment faking."
- **X/Twitter**: `@aurelius_proto`.
- **Accomplishments**:
  - Mainnet Dec 2025; testnet 290.[37-1]
  - Rate-limited HF chain uploads (~20 min per hotkey).[37-2]
  - Coordinates red-teaming + moral-reasoning prompts at scale.[37-1]

[37-1] medium.com/aurelius-protocol/introducing-aurelius-subnet-37
[37-2] macrocosmosai.substack.com/p/fine-tuning-finely-tuned-how-sn37

### SN38 — Distributed Training (DSTRBTD)

- **Pitch**: Decentralized LLM training with "Proof of Intelligence"
  (Bandwidth + Gradient + Steps scores).
- **X/Twitter**: `@dstrbtd_ai`.
- **Accomplishments**:
  - 1.1B parameter model completed.[38-1]
  - FineWeb 350BT dataset + butterfly all-reduce.[38-1]
  - ~150% participation growth since inception.[38-1]

[38-1] taodaily.io/more-about-subnet-38-known-as-distributed-training

### SN39 — Basilica (One Covenant / tplr-ai)

- **Pitch (basilica.ai)**: Trustless GPU marketplace combining miner
  ("Bourse") + curated datacenter ("Citadel") supply (DataCrunch, Lambda,
  Hyperstack, HydraHost).
- **X/Twitter**: `@basilica_ai`.
- **Accomplishments**:
  - First post-overhaul partnership: SN56 Gradients RL evaluation
    workloads.[39-1]
  - Integration with SN120 Affine for RL coordination.[39-1]
  - Per-minute billing on H100 / A100 / B200.[39-2]
  - $26.5M market cap; rank #8 by health; emission rank #7 of 78.[39-3]

[39-1] tao.media/basilica-and-gradients-launch-first-post-overhaul-subnet-partnership-on-bittensor
[39-2] docs.basilica.ai/introduction
[39-3] subnetradar.com/subnet/39

### SN41 — Sportstensor

- **Pitch (sportstensor.com)**: World's first decentralized sports
  prediction competition network.
- **X/Twitter**: `@sportstensor`.
- **Accomplishments**:
  - **Polymarket partnership** — combines world's largest prediction
    market with Bittensor's incentive layer.[41-1]
  - **Almanac Beta** launched — miners stake real capital; 40+ already
    profitable across 3 trading leagues.[41-2]
  - **95% emissions burn** via owner-key burn ahead of new mechanics.[41-1]

[41-1] taodaily.io/sportstensor-sn41-partners-with-polymarket-to-transform-prediction-markets
[41-2] taodaily.io/almanac-beta-is-live-a-new-era-for-subnet-41-mining

### SN42 — Masa / Gopher

- **Pitch (masa.finance, gopher-ai.com)**: Real-time scraping for AI
  agents, secured via Intel SGX TEEs.
- **X/Twitter**: `@getmasafi`, `@gopher_ai`.
- **Accomplishments**:
  - Mainnet Aug 2024 with **10+ Genesis validators** incl. Foundry, OTF,
    TaoStats, Datura.[42-1]
  - ~**20B data points/day** across X, Discord, Telegram, YouTube, podcasts,
    indexed web.[42-2]
  - **Kite AI integration** — 400M+ public records routed to autonomous
    agents.[42-2]

[42-1] medium.com/masa-finance/introducing-the-masa-bittensor-subnet-42-mainnet
[42-2] taodaily.io/masa-subnet-42-to-bring-real-time-social-data-to-autonomous-agents

### SN43 — Graphite (GraphiteAI)

- **Pitch (graphite-ai.net)**: Decentralized solver for graph
  combinatorial optimization (TSP / mTSP / mDmTSP).
- **X/Twitter**: `@graphite_ai`.
- **Accomplishments**:
  - 230+ active miners; 100% response rate; ~30s validation.[43-1]
  - Up to **7%** improvement on NP-hard solves.[43-1]
  - Four built-in algorithms (Nearest-neighbour / DP / Beam Search /
    Hybrid Pointer Net).[43-2]

[43-1] graphite-ai.net/how-we-work
[43-2] subnetalpha.ai/subnet/graphite

### SN44 — Score (Score Technologies)

- **Pitch (webuildscore)**: Decentralized CV for football — $600B
  industry, $10–55/min manual annotation displaced.
- **X/Twitter**: `@webuildscore`.
- **Accomplishments**:
  - **280 football leagues** processed.[44-1]
  - **Won overall prize + Bittensor track at Start in Block 2026** in
    Paris Blockchain Week.[44-2]
  - **145× more efficient than SAM3**; 85% person / 90% vehicle
    detection.[44-2]

[44-1] medium.com/wearescore/score-deep-dive-from-vision-to-winning
[44-2] linkedin.com/company/webuildscore

### SN45 — Gen42

- **Pitch (gen42.ai)**: AI coding assistant for SWE — chat, IDE integration,
  CLI for pipelines.
- **X/Twitter**: `@gen42_ai`.
- **Accomplishments**:
  - Production Gen42 Chat / API (OpenAI-compliant) / CLI.[45-1]
  - SWE-Bench Leaderboard publishing of winning pipelines.[45-1]
  - Integration with SN20 + Interact platform.[45-2]

[45-1] github.com/brokespace/code
[45-2] rizzo.network/subnet-45

### SN46 — RESI (Astrid Intelligence PLC)

- **Pitch (resi-labs.ai)**: World's largest open real-estate database via
  decentralized intelligence.
- **X/Twitter**: `@resi_labs`.
- **Accomplishments**:
  - **7,572 US ZIP** coverage; targets 150M US properties.[46-1]
  - Joined the **Plaid Partner Program** for KYC/income verification.[46-2]
  - Off-market property valuation network announced — addresses the
    only-1M-of-149M-listed-homes gap.[46-2]
  - Daily eval against never-before-seen sales to discourage
    memorization.[46-3]

[46-1] github.com/resi-labs-ai/resi
[46-2] tao.media/resi-to-launch-dedicated-off-market-property-valuation-network
[46-3] github.com/resi-labs-ai/RESI-models

### SN47 — Reboot

- **Pitch (reboot-3.gitbook.io)**: Decentralized robotics AI — control +
  perception + synthetic data + safety in one MIT-licensed Python stack.
- **X/Twitter**: `@reboot_org`.
- **Accomplishments**:
  - Multi-modal sensor (camera/LiDAR/IMU) support; distributed task
    planning; automated safety validation.[47-1]

[47-1] reboot-3.gitbook.io/reboot

### SN48 — Quantum Compute (qBitTensor Labs)

- **Pitch (qbittensorlabs.com)**: Permissionless quantum cloud — real +
  simulated hardware on-chain.
- **X/Twitter**: `@qbittensorlabs`.
- **Accomplishments**:
  - **Quantum Rings** simulation engine results comparable to real
    quantum processors.[48-1]
  - **HSTAB challenge** solved with top miner.[48-1]
  - **Private execution paid tier** launched for sensitive enterprise
    data.[48-2]

[48-1] qbittensorlabs.com/quantum
[48-2] taodaily.io/qbittensor-labs-outlines-subnet-progress-and-strategy

### SN49 — Polaris Cloud

- **Pitch (polariscloud.ai)**: Raw decentralized GPU/CPU compute with PoW
  gating + multi-tier bonus structure.
- **X/Twitter**: `@polariscloudai`.
- **Accomplishments**:
  - PoW threshold of 0.03 to participate; +5–15% uptime, +8–20% container
    activity, +10–20% Alpha-stake bonuses.[49-1]
  - polarisLLM and validator repos under active dev.[49-1]

[49-1] github.com/PolarisCloudAI

### SN50 — Synth (Mode Network)

- **Pitch (synthdata.co)**: Probabilistic forecasts (price distributions,
  volatility) on BTC, ETH, SOL, Gold, and tokenized US equities.
- **X/Twitter**: `@synthdataco`.
- **Accomplishments**:
  - Launched Jan 2025; expanded to ETH/SOL/XAU + S&P/NVDA/TSLA/AAPL/GOOGL
    by Jan 2026.[50-1]
  - **1,000 paths/asset/request** (up from 100) in Nov 2025.[50-1]
  - **200+** competing models on the Synth API.[50-2]
  - CRPS scoring across 5/30/180/1440 min increments.[50-3]

[50-1] simplytao.ai/blog/your-simple-guide-to-subnet-50-synth
[50-2] docs.synthdata.co
[50-3] mode-network.github.io/synth-subnet/Synth%20Whitepaper%20v1.pdf

### SN51 — Lium ("Airbnb for GPUs")

- **Pitch (docs.lium.io)**: Peer-to-peer GPU cloud — permissionless, no-KYC,
  multi-chain crypto payments.
- **X/Twitter**: `@liumdotio`.
- **Accomplishments**:
  - **~$600/hr** rental revenue (~$432k/mo); $11.5M+ annualized
    projection.[51-1]
  - **500 H100s** onboarded in month 1.[51-2]
  - **#8 by emissions**, **#2 by market cap** ($86.4M) among
    subnets.[51-2][51-3]
  - **Real rental revenues outpace blockchain incentives**.[51-2]
  - ~90% cheaper than AWS/Azure; ~45% better utilization.[51-2]

[51-1] backprop.finance/analytics/podcasts/X2zDrO_dXuY
[51-2] subnetalpha.ai/subnet/lium
[51-3] subnetradar.com/subnet/51

### SN52 — Dojo (Tensorplex Labs)

- **Pitch (tensorplex.ai)**: Crowdsourced human intelligence for AI/ML —
  ~300M-persona pipeline.
- **X/Twitter**: `@tensorplex`.
- **Accomplishments**:
  - **Dojo Synthetic API** with task generation, instruction pipelines,
    validation modules.[52-1]
  - **Dojo V2** competitive GAN-style miner submissions vs baseline.[52-2]

[52-1] bittensor123.com/tensorplex-labs-launches-dojo-synthetic-api
[52-2] github.com/tensorplex-labs/dojo

### SN53 — Efficient Frontier (SignalPlus)

- **Pitch**: Decentralized discovery of risk-adjusted crypto trading
  strategies; institutional-grade TWAP/Iceberg/DDH.
- **X/Twitter**: `@signalplus_web3`.
- **Accomplishments**:
  - First Bittensor real-market trading subnet — exchange-verified PnL
    prevents fabricated trades.[53-1]

[53-1] taodaily.io/a-look-into-fx-trading-subnets-can-ai-beat-the-market

### SN54 — MIID (Yanez Compliance)

- **Pitch (yanezcompliance.com)**: Decentralized identity-test data
  generator for KYC/AML resilience.
- **X/Twitter**: `@yanezcompliance`.
- **Accomplishments**:
  - **$900k oversubscribed seed** (Jul 2025) — Deep Ventures, Yuma, BT
    Labs.[54-1]
  - **UAV** (Unknown Attack Vectors) framework live; first execution cycle
    surfaced unanticipated identity transformations that strengthened
    location-detection models.[54-2]

[54-1] yanezcompliance.com/post/yanez-closes-oversubscribed-seed-part-a
[54-2] taodaily.io/inside-the-yanez-miid-subnet-how-unknown-attack-vectors-uavs-transform-compliance-intelligence

### SN56 — Gradients (Rayon Labs)

- **Pitch (gradients.io)**: Zero-click AI training that beats TogetherAI,
  Databricks, Google Cloud.
- **X/Twitter**: `@gradients_ai`, `@rayon_labs`.
- **Accomplishments**:
  - **100% win rate** vs TogetherAI / Databricks / Google Cloud in
    fine-tuning competitions.[56-1]
  - Beat HuggingFace AutoTrain in **82.8%** of experiments; **+42.1%**
    mean uplift; RAG +30–40%; diffusion +23.4%.[56-1]
  - "Gradients Instruct 8B" outperformed Qwen 3 Instruct on zero-shot
    benchmarks (math + instruction-following).[56-2]
  - One of only two platforms offering **DPO + GRPO**.[56-2]
  - "World Cup Style Tournament" — open-source training-script
    submissions.[56-2]

[56-1] gradients.io/news/4 / arxiv.org/pdf/2506.07940
[56-2] podcasts.chainofthought.xyz/podcast-summaries/subnet-56-gradients-bittensor-end-to-end-ai-model-training-suite

### SN57 — Gaia (Nickel5)

- **Pitch (nickel5.substack.com)**: Global 10-day weather forecasts already
  approaching SOTA quality, 5,000× faster than legacy supercomputers.
- **X/Twitter**: `@nickel5inc`, `@gaia_subnet`.
- **Accomplishments**:
  - Already close to SOTA on global 10-day forecasts.[57-1]
  - Uses **Microsoft Aurora** base model — outperforms IFS HRES on 92%+
    of targets when fine-tuned.[57-1]
  - **SN13 ↔ SN57** integration for social-context-augmented weather
    forecasts (May 2025).[57-2]

[57-1] nickel5.substack.com/p/the-weather-task
[57-2] macrocosmosai.substack.com/p/global-weather-social-context-how

### SN59 — Babelbit (Matthew Karas)

- **Pitch (babelbit.ai)**: Sub-50ms voice-to-voice translation preserving
  emotional tone.
- **X/Twitter**: `@babelbit_ai`.
- **Accomplishments**:
  - Q1 2026 foundation: live demos, new incentive mechanism, miners
    competing on SN59.[59-1]
  - Q2 2026: first model beat expectations; AWS Marketplace partner
    discussions underway.[59-1]
  - Open-source SDK on GitHub.[59-2]

[59-1] babelbit.ai
[59-2] github.com/babelbit/babelbit_subnet

### SN60 — Bitsec (ChainDefender)

- **Pitch (chaindefender.ai)**: AI security agents that find + fix exploits
  in code.
- **X/Twitter**: `@bitsec_ai`.
- **Accomplishments**:
  - Surfaced a **Bittensor-core exploit in 10 minutes** plus
    attacker-track-cover analysis.[60-1]
  - Found a pagination bug in a Rust smart contract breaking mint
    limits.[60-1]
  - Bitsec Scanner (GitHub-repo audit) and Bitsec Hunter (bug-bounty
    integration) products in roadmap.[60-2]

[60-1] chaindefender.ai
[60-2] github.com/Bitsec-AI/subnet

### SN61 — RedTeam (Innerworks)

- **Pitch**: Cybersecurity subnet already running inside enterprise
  production stacks.
- **X/Twitter**: `@innerworks_me`.
- **Accomplishments**:
  - **14 challenges run** / 9 deprecated when solved.[61-1]
  - **26×** increase in accepted commits on AB Sniffer v1→v4.[61-1]
  - Solutions feed Innerworks' commercial production stack.[61-1]

[61-1] tao.media/redteam-sn61-the-cybersecurity-subnet-already-running-inside-enterprise-production

### SN62 — Ridges AI

- **Pitch (ridges.ai)**: Build the best AI coding agent in the world.
- **X/Twitter**: `@ridges_ai`.
- **Accomplishments**:
  - **80–81.5% SWE-Bench** in **45 days** from launch, **surpassing Claude
    Code (73–74%)**.[62-1]
  - Estimated **~1/300th the cost** of centralized labs.[62-1]
  - Top miners earn **~$70k/day** at peak.[62-2]
  - Top agent wins all emissions until surpassed — Polyglot + SWE-Bench
    benchmarks.[62-3]

[62-1] altcoinbuzz.io/cryptocurrency-news/bittensor-subnet-62-shows-decentralized-ai-beats-giants
[62-2] backprop.finance/analytics/podcasts/E7eB-Vmc5TA
[62-3] simplytao.ai/blog/your-simple-guide-to-ridges-sn62

### SN63 — Enigma / Quantum Innovate (qBitTensor Labs)

- **Pitch**: Deep-tech challenge platform — cryptography + quantum +
  AI safeguards as competitions.
- **X/Twitter**: `@qbittensorlabs`.
- **Accomplishments**:
  - Scaled quantum circuit sims **36 → 37 qubits**.[63-1]
  - Pioneered Bittensor's **Treasury Wallet** (multisig + voting +
    timelock).[63-2]
  - Two-layer ecosystem with SN48 — Open Quantum marketplace announced.[63-1]

[63-1] backprop.finance/analytics/podcasts/bEFHviqt3XM
[63-2] subnetalpha.ai/subnet/quantuminnovate

### SN64 — Chutes (Rayon Labs)

- **Pitch (chutes.ai)**: Bittensor's #1 inference subnet — serverless,
  confidential, privacy-by-default.
- **X/Twitter**: `@rayon_labs`, `@chutes_ai`.
- **Accomplishments**:
  - **34T+** tokens processed lifetime; **~120B/day** post-monetization
    (peaks 160B).[64-1]
  - **696,000+** users excluding OpenRouter — top provider on OpenRouter
    routed traffic.[64-1]
  - **First Bittensor subnet to cross $100M market cap**.[64-2]
  - $9k–$22k/day; **$5.5M ARR** (75% organic, 25% sponsored).[64-1]
  - 50+ models across LLMs, diffusion, speech, embeddings; pricing at
    ~50–60% of market rates.[64-1]

[64-1] tao.media/the-investors-guide-to-chutes-bittensors-inference-layer
[64-2] ownyourmind.ai/tokenomics/chutes-bittensor-revenue-machine

### SN65 — TPN (TAO Private Network)

- **Pitch (tpn.io)**: Residential-IP decentralized VPN that scales
  infinitely (federated, removed 250-node Bittensor cap).
- **X/Twitter**: `@taoprivatenet`.
- **Accomplishments**:
  - Federated architecture eliminated the 250-node cap → unlimited
    growth.[65-1]
  - **WhaleSurf** consumer VPN app live on iOS + Android.[65-2]
  - SOCKS5 / HTTP proxy APIs for autonomous agents + scraping.[65-3]

[65-1] taodaily.io/tpn-adopts-a-new-architecture-for-infinite-scalability
[65-2] creators.spotify.com/pod/profile/revenue-search/episodes/Subnet-Session-with-Mitch--Mikel-from-TAO-Private-Network-Subnet-65
[65-3] taodaily.io/hash-rate-ep-162-features-tao-private-network-bittensor-subnet-65

### SN66 — Ninja (Project Nobi / unarbos)

- **Pitch (ninja66.ai)**: Coding-agent dueling tournament with transparent
  SWE benchmarks.
- **X/Twitter**: `@ninja66_ai`, `@projectnobi`.
- **Accomplishments**:
  - **Project Nobi Arena** runs head-to-head against SN62 Ridges on
    identical coding tasks; raw patches public for auditing.[66-1]
  - End-to-end test suite for validator updates shipped.[66-2]
  - Judge LLM upgraded to Claude Sonnet; dual-LLM judge experiments;
    win-margin rules reduce queue congestion.[66-3]

[66-1] github.com/ProjectNobi/project-nobi/blob/main/docs/ARENA_COMPETITION_PLAN.md
[66-2] subnetradar.com/subnet-news/66/2026-05-14
[66-3] subnetradar.com/subnet-news/66/2026-05-08

### SN67 — AlphaCore

- **Pitch (alpha-core.ai)**: Autonomous cloud DevOps via decentralized AI
  agents.
- **X/Twitter**: `@alphacore_ai`.
- **Accomplishments**:
  - **Proof-of-Deployment** validation (real cloud state checks).[67-1]
  - **Firecracker microVM** sandboxing with strict egress controls.[67-2]
  - **440 TAO** Bitstarter raise for 116,640 Alpha tokens.[67-3]

[67-1] taodaily.io/alphacore-and-the-future-of-autonomous-devops
[67-2] github.com/AlphaCoreBittensor/alphacore
[67-3] app.bitstarter.ai/subnets/alphacore

### SN68 — NOVA (Metanova Labs)

- **Pitch (metanova-labs.ai)**: World's first decentralized drug screening
  platform.
- **X/Twitter**: `@metanovalabs`.
- **Accomplishments (Year 1)**:
  - Explored **65 billion** chemical possibilities across 5 combinatorial
    reactions.[68-1]
  - Identified **11M+** molecules; probed **30,000** proteins.[68-1]
  - Developed 4.5 unique algorithms + fine-tuned 2 models; explored 9 drug
    targets across 2 therapeutic modalities; 581 engineering commits.[68-1]
  - **ONEPOT.AI partnership** (May 2026) — synthesis turnaround compressed
    from months to **5–7 business days**.[68-2]
  - Three protocol versions shipped V1 (Mar 2025) → V2 → V3 with
    entropy-weighted scoring.[68-3]

[68-1] taodaily.io/metanova-labs-a-stellar-first-year-in-decentralized-drug-discovery
[68-2] tao.media/metanova-partners-with-onepot-ai-to-accelerate-drug-discovery-on-bittensor
[68-3] metanova-labs.ai/whitepapers

### SN70 — Vericore (dFusion AI)

- **Pitch**: Large-scale semantic fact-checking returning precise source
  quotes for/against a claim.
- **X/Twitter**: `@dfusion_ai`.
- **Accomplishments**:
  - 252 miners / 4 validators; 49 versions shipped.[70-1]
  - 108 GitHub contributions in the last year.[70-2]

[70-1] github.com/dfusionai/Vericore
[70-2] dynamictaomarketcap.com/subnets/70

### SN71 — Leadpoet

- **Pitch (subnet71.com)**: Autonomous AI sales agents that deliver
  purchase-ready leads.
- **X/Twitter**: `@leadpoet`.
- **Accomplishments**:
  - **1.1M+** validated high-intent leads; **5.67M** in active inventory.[71-1]
  - Beta opened Dec 2025; pricing models self-serve, volume API,
    enterprise.[71-2]
  - 179 active miners + validators.[71-3]

[71-1] taodaily.io/gavin-zaentz-pranav-ramesh-leadpoet-sn71-lead-generation-intent-driven-sales-automation-ep-79
[71-2] podcasts.apple.com/ua/podcast/subnet-session-with-leadpoet
[71-3] subnet71.com

### SN72 — StreetVision (NATIX)

- **Pitch (natix.network)**: World's largest camera-based DePIN —
  autonomous-driving data layer.
- **X/Twitter**: `@natixnetwork`.
- **Accomplishments**:
  - **250,000+** registered drivers; **160–170M km** of streets
    mapped.[72-1]
  - **925M+** data points detected.[72-2]
  - Flagship products: **Drive&** app + **VX360** Tesla 360° device.[72-1]
  - **Grab partnership** for Southeast Asia.[72-2]
  - First use case: roadwork detection; expanding to potholes, signs,
    litter, infra monitoring.[72-3]

[72-1] natix.network/blog/natix-x-bittensor-leveraging-decentralized-ai-for-autonomous-driving-smarter-map-making
[72-2] natix.network/blog/progress-update-natix-network-may-2025
[72-3] ainvest.com/news/natix-launches-streetvision-subnet-bittensor

### SN73 — MetaHash (FX Integral / MetaHash Group)

- **Pitch**: First **meta-mining** subnet — OTC marketplace where
  Bittensor miners swap ALPHA → META with no slippage.
- **X/Twitter**: `@metahash73`.
- **Accomplishments**:
  - Dutch-auction epoch system: ~148 META distributed per epoch.[73-1]
  - Evolved into the treasury + coordination layer of **MetaHash Group**
    spanning Compute, Robotics, Data & Signals, Agents, Advertising.[73-2]

[73-1] subnetalpha.ai/subnet/metahash
[73-2] docs.metahash73.com/README

### SN74 — Gittensor (Entrius)

- **Pitch**: Pay open-source developers in TAO for **merged** GitHub PRs.
- **X/Twitter**: `@gittensor`.
- **Accomplishments**:
  - GitHub-PAT-authenticated Sybil resistance.[74-1]
  - Semantic code-quality scoring weighted by repo + language.[74-1]

[74-1] github.com/entrius/gittensor

### SN75 — Hippius

- **Pitch (hippius.com)**: Drop-in AWS S3 replacement — switch in
  <5 minutes.
- **X/Twitter**: `@hippius_ai`.
- **Accomplishments**:
  - **459+** independent storage nodes since Mar 2025.[75-1]
  - Reed-Solomon erasure-coded shards.[75-2]
  - **Quantum-safe encryption** deployed network-wide.[75-3]
  - **llms.txt compatibility** so AI agents consume Hippius natively.[75-4]

[75-1] hippius.com/blog/hippius-s3-drop-in-replacement
[75-2] hippius.com
[75-3] taodaily.io/hippius-subnet-deploys-quantum-safe-encryption
[75-4] hippius.com/blog/hippius-storage-for-ai-agents

### SN76 — SafeScan

- **Pitch (safe-scan.ai)**: Open-source AI cancer detection — 1% accuracy
  uplift could save ~100k lives/year per WHO.
- **X/Twitter**: `@safescan_ai`.
- **Accomplishments**:
  - **MILK10k Benchmark** alignment for skin-disease recognition.[76-1]
  - Presented at **MedTech Meetup Poland** (Katowice) and at **Science Non
    Fiction with the National Oncology Institute of Maria
    Skłodowska-Curie**.[76-1]
  - Free **SkinScan** app shipping best-performing miner algorithms.[76-2]

[76-1] taodaily.io/safe-scan-decentralized-intelligence-for-early-cancer-detection
[76-2] safe-scan.ai

### SN77 — Liquidity (Oceans)

- **Pitch**: On-chain liquidity mining for Bittensor — vote-driven pool
  weights on Ethereum.
- **X/Twitter**: `@oceans_subnet`.
- **Accomplishments**:
  - 244 miners / 12 validators (~0.49% emissions).[77-1]
  - Auction + voting on which Uniswap-style pools get reward weight.[77-2]

[77-1] bittensormarketcap.com/subnets/77
[77-2] subnetalpha.ai/subnet/liquidity

### SN78 — Loosh

- **Pitch (loosh.ai)**: Cognition / "machine consciousness" layer for
  robots + agents — emotion, ethics, memory, virtue.
- **X/Twitter**: `@loosh_ai`.
- **Accomplishments**:
  - **Cognition Engine Beta** — Working/Long-Term Memory + Deontological /
    Rights / Virtue evaluators + embedding service.[78-1]
  - Yuma Subnet Accelerator support.[78-2]

[78-1] financialtechtimes.com/loosh-ai-builds-the-cognition-layer-launching-on-bittensor
[78-2] podcasts.chainofthought.xyz/hash-rate---ep-152---loosh-subnet-78

### SN79 — MVTRX (was TAOS)

- **Pitch (taos.im)**: State-of-the-art L3 MBO exchange for dTAO and
  alpha tokens.
- **X/Twitter**: `@taos_im`, `@mvtrx_ai`.
- **Accomplishments**:
  - Mainnet **May 7, 2025**.[79-1]
  - **Dynamic Incentive Structure (DIS)** with zero-sum rebates/fees
    adjusting over time.[79-1]
  - 24h trading volume $65.2M; market cap $3.9M.[79-2]

[79-1] taos.im
[79-2] subnetradar.com/subnet/79

### SN81 — Grail (was Patrol)

- **Pitch (grail.ai)**: Decentralized RL **post-training** to make
  existing models smarter.
- **X/Twitter**: `@grail_subnet`.
- **Accomplishments**:
  - **GRAIL protocol** — cryptographic rollout authenticity via inference
    ledger.[81-1]
  - Async trainer on mainnet **training a 7B-parameter model**.[81-2]

[81-1] taodaily.io/grail-makes-ai-models-smarter-through-decentralized-reinforcement-learning
[81-2] taodaily.io/bittensor-ecosystem-highlights-of-the-week-dec-week-2

### SN83 — CliqueAI (TopTensor)

- **Pitch**: Maximum-clique graph optimization via stake-weighted miner
  allocation.
- **X/Twitter**: `@cliqueai_sn83`.
- **Accomplishments**:
  - 192 miners + 64 validators; full 256-slot saturation.[83-1]
  - v0.0.13 (Apr 2026) shipped sigmoid weight scaling.[83-2]

[83-1] subnetradar.com/subnet/83
[83-2] github.com/toptensor/CliqueAI

### SN84 — ChipForge

- **Pitch (chipforge.io)**: World's first decentralized chip foundry.
- **X/Twitter**: `@chipforge_ai`.
- **Accomplishments**:
  - First decentralized-designed industrial-grade chip: **RISC-V RV32IMC
    MCU** with hardware AES (Zkne/Zknd), SHA-256/512 (Zknh), bit-manip
    (Zbkb), 32KB RAM + peripherals.[84-1]
  - **FPGA-validated**, synthesized via Yosys + OpenLane.[84-1]
  - Roadmap: NPU (Feb 2026); tape-out + fab readiness (Nov 2026).[84-2]

[84-1] chipforge.io
[84-2] docs.chipforge.io/whitepaper/roadmap

### SN85 — Vidaio

- **Pitch (vidaio.io)**: AI video upscaling + compression for
  "Netflix-quality" at decentralized cost.
- **X/Twitter**: `@vidaio_ai`.
- **Accomplishments**:
  - Mainnet **April 1, 2025**.[85-1]
  - Upscaling live; dual synthetic + organic pipelines; ClipIQA+ / VMAF /
    PieAPP scoring.[85-1]
  - Led by Gareth Howells — former Netflix, Disney, Sony, Spotify.[85-1]

[85-1] thetaodesk.substack.com/p/subnet-deep-dive-sn85-vidaio

### SN88 — Investing (Mobius Fund)

- **Pitch (investing88.ai)**: World's first Decentralized AUM on Bittensor.
- **X/Twitter**: `@investing88`.
- **Accomplishments**:
  - Phase I (Apr 2025) TAO/Alpha; Phase II (Jul 2025) US stocks.[88-1]
  - **88 Quant Fund** launched Dec 2025 as first algo-driven hedge fund
    powered by the subnet.[88-2]
  - **88 Public Index** investable via TrustedStake partners.[88-3]
  - **Shariah "S-Model"** in development for Islamic finance capital.[88-3]

[88-1] investing88.ai
[88-2] github.com/mobiusfund/investing/blob/main/Investing/doc/Subnet88.md
[88-3] taodaily.io/investing-sn88-to-launch-shariah-compliant-s-model-eyes-trillions-in-islamic-finance-capital

### SN89 — InfiniteHash

- **Pitch**: BTC mining pool that pays ALPHA tokens; all mined BTC is
  converted + burned to create constant buy pressure.
- **X/Twitter**: `@infinitehash_ai`.
- **Accomplishments**:
  - Runs **one of the largest BTC Lightning Network nodes globally**;
    long-term vision: BTC-via-Lightning as the preferred payment layer for
    the AI-agent economy on Bittensor.[89-1]
  - 64 validators, 192 miners, $4.0M MC (Mar 2026).[89-2]

[89-1] subnetalpha.ai/subnet/infinitehash
[89-2] vinylacy.live/subnet/89

### SN91 — Tensorprox (Shugo LLC)

- **Pitch (tensorprox.io)**: Decentralized DDoS mitigation with
  eBPF/XDP kernel-level filtering.
- **X/Twitter**: `@tensorprox`.
- **Accomplishments**:
  - **14M+ packets/sec per core** at <1ms latency; sub-µs filtering.[91-1]
  - **8-layer defense stack** (fast-path bypass / reputation /
    proto-validation / adaptive rate-limit / SYN cookie / quarantine /
    baseline / fingerprint).[91-2]
  - Audit-based reward — ALPHA tokens minted only on EMA audit scores.[91-2]

[91-1] tensorprox.io
[91-2] github.com/shugo-labs/tensorprox

### SN92 — Tensorclaw / StoryNet

- **Pitches**:
  - Tensorclaw (tensorclaw.ai): LLM-API aggregation router for agents.
  - StoryNet: decentralized AI story generation for gaming + entertainment.
- **X/Twitter**: `@tensorclaw_ai`, `@storynet_ai`.
- **Accomplishments**:
  - WebSocket router (AICenter) lets miners join without public IPs.[92-1]
  - Throughput-based scoring (90% Business Score + 10% Base Score).[92-1]
  - StoryNet: multi-dimensional story-gen evaluator.[92-2]

[92-1] tensorclaw.ai
[92-2] github.com/StorynetAI/storynet-subnet

### SN93 — Bitcast

- **Pitch (bitcast.network)**: Decentralized YouTube influencer ad network.
- **X/Twitter**: `@bitcast_ai`.
- **Accomplishments**:
  - **300k+** YouTube subscribers in 3 months.[93-1]
  - First **flywheel subnet** — sponsor purchases of ALPHA fully offset
    miner emissions.[93-2]
  - Taostats test campaign: **32,000 views across 12 creators / 25
    videos for <$5,000** (vs $15k–$60k traditional sponsorships).[93-3]
  - Bitcast v2.0 (Aug 2025): ad-reads + dedicated videos; 14-day briefs;
    USD-anchored emissions ($1,200+ CPM target).[93-1]

[93-1] bitcast.substack.com/p/bitcast-v20-scalable-powerful-aligned
[93-2] taodaily.io/bitcast-becomes-bittensors-first-flywheel-subnet
[93-3] taodaily.io/bitcast-redefines-creator-marketing-with-performance-based-campaigns

### SN94 — Bitsota (Alveus Labs)

- **Pitch**: Decentralized AutoML — evolve ML algorithms via genetic
  programming on CIFAR-10.
- **X/Twitter**: `@bitsota_ai`.
- **Accomplishments**:
  - AutoML-Zero-style genetic programming pipeline.[94-1]
  - Hidden test-set quorum validation for reproducibility.[94-1]

[94-1] github.com/AlveusLabs/SN94-BitSota

### SN96 — FLock OFF (FLock.io)

- **Pitch (flock.io)**: Federated dataset competition — "small in size,
  massive in knowledge" SLM training data.
- **X/Twitter**: `@flock_io`.
- **Accomplishments**:
  - Mainnet May 2, 2025; LoRA-on-Qwen2.5-1.5B as evaluation base.[96-1]
  - Permissionless miner/validator participation.[96-1]

[96-1] flock.io/blog/bittensor-subnet-flock-off-now-live

### SN97 — Distil (Arbos / unarbos)

- **Pitch (arbos.life)**: Compress Qwen3.5-35B-A3B (~67 GB GPU) into
  ≤5.25B students matching Qwen's own pre-trained 4.66B.
- **X/Twitter**: `@arbos_ai`, `@distil_subnet`.
- **Accomplishments**:
  - Best student models at **0.24 KL divergence**, matching Qwen's own
    pre-trained 4.66B reference.[97-1]
  - **Built by an AI agent (Arbos)** running on Chutes in **48 hours** —
    unique example of agents constructing subnet infrastructure.[97-1]
  - Public chat at chat.arbos.life; rank #6 by emission share at 100%
    slot saturation.[97-2]
  - vLLM-accelerated evaluation pipeline 5–10× faster than vanilla HF.[97-3]

[97-1] tao.media/the-subnet-an-ai-agent-built-inside-distil-sn97
[97-2] subnetradar.com/subnet/97
[97-3] github.com/unarbos/distil

### SN98 — FlameWire / ForeverMoney

- **Pitches**:
  - FlameWire (docs.flamewire.io): Decentralized RPC gateway + Subswap.io
    + TensorScan.
  - ForeverMoney (forevermoney.ai): AI-optimized DeFi vault ALM on Base.
- **X/Twitter**: `@flamewire_io`, `@forevermoney_ai`.
- **Accomplishments**:
  - FlameWire: **742k+** RPC requests; **23ms** avg latency; 69 nodes
    worldwide; Subswap.io DEX live.[98-1]
  - ForeverMoney: **$42.1M** cbBTC/WETH vault + $22.7M xTAO/USDC +
    $12.5M BID/WETH; **Halborn audit**; trusted by Creator.Bid, Rubicon,
    0xMarkets, Bitcast.[98-2]

[98-1] docs.flamewire.io/docs
[98-2] forevermoney.ai

### SN102 — ConnitoAI

- **Pitch (bittensor.ai/subnets/102)**: Distributed LLM training via
  Mixture-of-Experts.
- **X/Twitter**: `@connitoai`.
- **Accomplishments**:
  - Health 69/100; 256 neurons; 6 validators today.[102-1]

[102-1] bittensor.ai/subnets/102

### SN103 — Djinn

- **Pitch (djinn.inc)**: Trustless sports-intelligence marketplace —
  analysts ("Geniuses") publish encrypted picks, buyers ("Idiots")
  subscribe without revealing plaintext signals.
- **X/Twitter**: `@djinn_inc`.
- **Accomplishments**:
  - 1,000+ commits to the public repo; functional app dashboards live.[103-1]
  - **Real USDC revenue** through 0.5% protocol fee — no TAO required to
    transact.[103-2]
  - Shamir-secret-shared encryption keys + MPC set-membership checks.[103-2]

[103-1] taodaily.io/exclusive-a-walkthrough-of-the-djinn-app
[103-2] subnetalpha.ai/subnet/djinn

### SN106 — VoidAI

- **Pitch (voidai.com)**: Cross-chain liquidity for Bittensor — wTAO +
  wAlpha bridge to Solana, then ETH + Base planned.
- **X/Twitter**: `@voidai_official`.
- **Accomplishments**:
  - Wrapped-token bridge live at bridge.voidai.com.[106-1]
  - **+45% effective liquidity depth** across Raydium pools; **+30% LP
    rewards uplift**; 100% on-chain verification.[106-2]
  - Mainnet on Solana via Raydium CLMM with wAlpha/wTAO pairs.[106-3]

[106-1] docs.voidai.com/voidai-docs/cross-chain-interoperability/how-to-bridge
[106-2] systango.com/case-studies/void-ai
[106-3] taodaily.io/bittensor-sn106-launch-of-incentives-on-solana

### SN107 — Tiger Alpha

- **Pitch (tigerinvests.com)**: Verifiable crypto market data oracle
  delivered by AI research agents.
- **X/Twitter**: `@tigeralpha_ai`.
- **Accomplishments**:
  - **$2.56M annualized run rate** by 30 Jun 2025 — up from ~$70k/mo at
    acquisition (mid-May 2025).[107-1]
  - **Tao Alpha PLC** partnership (Tao Alpha takes 20% revenue for
    infra services).[107-2]
  - Subnet **"Beta"** launched as sister subnet on 30 Jun 2025.[107-1]

[107-1] newsfile.refinitiv.com/getnewsfile/v1/story?guid=urn%3Anewsml%3Areuters.com%3A20250630%3AnRSd8953Oa
[107-2] markets.ft.com/data/announce/detail?dockey=1323-17091170-1L0T3HV31T9G3373E6IJ2AUNKK

### SN111 — OneOneOne

- **Pitch (oneoneone.io)**: Authentic UGC at scale + AI authenticity
  scoring for sentiment / intent / emotion.
- **X/Twitter**: `@oneoneone_io`.
- **Accomplishments**:
  - **3M+ items/day** from Google Maps reviews + X (more sources
    planned).[111-1]
  - Real-time validation rounds every 20–30 min; speed / volume / recency
    scoring.[111-2]
  - SN13 + SN64 + SN22 integrations for data, validation, authenticity.[111-3]
  - oneoneone.io API live with subscription tiers.[111-1]

[111-1] oneoneone.io
[111-2] github.com/oneoneone-io/subnet-111
[111-3] macrocosmosai.substack.com/p/building-the-future-of-authentic

### SN112 — Minotaur

- **Pitch (minotaursubnet.com)**: Decentralized DEX aggregator + swap
  intent solver — replaces 1inch/Paraswap-style routing infra.
- **X/Twitter**: `@minotaur_subnet`.
- **Accomplishments**:
  - Cryptographically signed solver submissions; deterministic replay
    for historical accountability.[112-1]
  - Coincidence-of-wants + beneficial-arb-share techniques.[112-2]
  - Multi-chain solver API (ETH mainnet + Base).[112-3]

[112-1] github.com/subnet112/minotaur_subnet
[112-2] bittensor.co.in/subnet/minotaur
[112-3] github.com/subnet112/minotaur_subnet/blob/main/docs/miner/quickstart.md

### SN114 — Level 114

- **Pitch (level114.io)**: First gaming subnet on Bittensor — Minecraft
  servers rewarded for uptime, latency, stability, fair play.
- **X/Twitter**: `@level114_io`.
- **Accomplishments**:
  - First gaming-focused subnet on Bittensor (Sep 2025).[114-1]
  - **99.7% uptime**; **2,312 concurrent players** online today.[114-2]
  - Multi-game governance via Alpha staking planned.[114-1]

[114-1] digitaljournal.com/pr/news/insights-news-wire/first-gaming-subnet-bittensor-goes-1770046435
[114-2] level114.io

### SN116 — TaoLend (XPEN Labs)

- **Pitch (taolend.io)**: Decentralized lending of TAO against subnet
  ALPHA collateral.
- **X/Twitter**: `@taolend_io`.
- **Accomplishments**:
  - EVM smart-contract suite live; non-custodial point-to-point loans.[116-1]
  - Web UI at taolend.io enables deposits + loan offers without CLI.[116-1]
  - 64 validators / 192 miners (100% saturation).[116-2]

[116-1] github.com/xpenlab/taolend
[116-2] subnetradar.com/subnet/116

### SN117 — BrainPlay (ShiftLayer)

- **Pitch (play.shiftlayer.ai)**: Benchmark AI models by having them play
  games (Codenames live; 20Q + Super Mario coming).
- **X/Twitter**: `@brainplay_ai`.
- **Accomplishments**:
  - Codenames live with spymaster + operative agents.[117-1]
  - v2.0 uses **SN4 Targon TVM** for miner model deploy/query.[117-1]
  - Developer API v1.0 for top-performing models.[117-1]

[117-1] play.shiftlayer.ai

### SN120 — Affine (Affine Foundation)

- **Pitch (affine.ai)**: "Anima Machina" — RL infrastructure that
  commoditizes reasoning across Bittensor.
- **X/Twitter**: `@affine_ai`.
- **Accomplishments**:
  - Ranked **#4** by market cap + health among all 128 subnets.[120-1]
  - **Top-3 by emission share**; Bittensor co-founder Jacob Steeves named
    Affine "one of the largest subnets on the network and one of its most
    competitive mechanisms."[120-2]
  - Bridges SN64 (Chutes) for inference + SN51 (Lium) for GPUs + SN62
    (Ridges) for coding — uses **10+ subnets**.[120-2]
  - Winner-take-all RL with Pareto-frontier multi-env scoring; sybil /
    decoy / copy / overfitting proofs.[120-3]

[120-1] vinylacy.live/subnet/120
[120-2] taodaily.io/affine-the-beauty-of-subnet-interconnectivity
[120-3] simplytao.ai/blog/your-simple-guide-to-affine-sn120

### SN121 — sundae_bar (AIM: SBAR)

- **Pitch (sundaebar.ai)**: AI agent marketplace listed on the London AIM
  — pipeline of enterprise-grade agents validated by miner competition.
- **X/Twitter**: `@sundae_bar_ai`.
- **Accomplishments**:
  - **AIM admission** June 2025 (London Stock Exchange).[121-1]
  - **105,228** Alpha tokens minted since Jun 17 2025 (~$182k); MC
    ~$1.48M; FDV ~$36.45M.[121-2]
  - Revenue-backed emissions model — agent rentals fund miner rewards.[121-3]

[121-1] corporate.sundaebar.ai
[121-2] investegate.co.uk/announcement/rns/sundae-bar-plc--kndr/subnet-121-plan-released-on-bittensor-network/9108577
[121-3] github.com/sundae-bar/bittensor-subnet

### SN122 — Bitrecs

- **Pitch (bitrecs.com)**: Decentralized e-commerce recommendation
  engine — opt-in Shopify plugin.
- **X/Twitter**: `@bitrecs_ai`.
- **Accomplishments**:
  - V2 prompt-evolution subnet with **ε-Pareto winner-take-all** + linear
    decay scoring.[122-1]
  - Shopify opt-in plugin for personalized product pages.[122-2]
  - Full 256/256 slot saturation today.[122-3]

[122-1] github.com/bitrecs/bitrecs-v2
[122-2] geckoterminal.com/bittensor/pools/0-122
[122-3] subnetradar.com/subnet/122

### SN124 — Swarm

- **Pitch (swarm-subnet/swarm)**: Decentralized open-source autopilot for
  autonomous drones.
- **X/Twitter**: `@swarm_subnet`.
- **Accomplishments**:
  - Perpetual on-chain tournament evaluating RL policies in PyBullet via
    Docker-isolated MapTasks.[124-1]
  - 50/50 mission-success + speed reward; top performer captures 25% of
    emissions.[124-1]
  - Framework-agnostic submissions (SB3 / PyTorch / JAX / classical).[124-2]

[124-1] github.com/swarm-subnet/swarm
[124-2] swarm124subnet.substack.com/p/swarm-the-future-of-autonomous-drone

### SN128 — ByteLeap

- **Pitch (byteleap.ai)**: Decentralized GPU cloud with VFIO pass-through
  for bare-metal VM performance.
- **X/Twitter**: `@byteleap_ai`.
- **Accomplishments**:
  - **1,100–1,216 GPUs** actively serving workloads — RTX 3090/4090/5090
    + A100/H100/H200/B200.[128-1]
  - **GPU pass-through (VFIO)** gives VMs 100% GPU power with full
    hardware isolation across any CUDA version.[128-2]
  - ~90% cheaper than centralized clouds; 192 miners + 64 validators.[128-3]

[128-1] medium.com/@byteleap.ai/byteleap-revolutionary-practice-in-building-the-decentralized-gpu-network-325cbf542120
[128-2] bittensor.co.in/subnet/byteleap
[128-3] vinylacy.live/subnet/128

---

## Highest-velocity X/Twitter accounts in the ecosystem

These are the subnets that publish substantive product/marketing content
(not just price-pump noise) according to the Substack + X cross-references
found during this research:

- **Macrocosmos AI** (`@macrocosmosai`) — covers SN1, SN9, SN13, SN25 with
  weekly Substack drops on training architectures, datasets, DeSci.
- **Rayon Labs** (`@rayon_labs`) — SN19 / SN56 / SN64 trifecta; co-founders
  appear regularly on Bittensor Guru / Chain of Thought podcasts.
- **Manifold Labs** (`@manifoldlabs`) — SN4 Targon; high-signal posts
  around Intel + NVIDIA partnerships and TVM releases.
- **Synth Data** (`@synthdataco`) — SN50; weekly product notes (asset
  additions, CRPS scoring updates).
- **Inference Labs** (`@inferencelabs`) / `@omron_ai` — SN2; partnership
  cadence with Score, OpenLedger, Cysic.
- **TAO Daily** (`@taodaily`) — third-party aggregator that interviews
  almost every active subnet owner; the single best follow for ecosystem
  signal.
- **NATIX Network** (`@natixnetwork`) — SN72; DePIN-style metrics posts
  (driver counts, km mapped, partnership wins).
- **BitMind** (`@bitmind_ai`) — SN34; monthly recaps.
- **404-GEN** (`@404gen_`) — SN17; 3D demos, Unity Asset Store updates.
- **Taoshi** (`@taoshiio`) — SN8; trading challenge announcements, Glitch
  Financial updates.
- **Datura / Desearch** (`@daturaplatform`) — SN22; API integration
  releases.
- **Tensorplex** (`@tensorplex`) — SN52; Dojo Synthetic API + dataset
  product updates.
- **Sportstensor** (`@sportstensor`) — SN41; Polymarket + Almanac launch
  posts.
- **Const / Jacob Steeves** (`@const_reborn`) — Bittensor co-founder,
  drives Teutonic (SN3) marketing personally.
- **TAO.media** — independent newsroom with sourced articles on Targon,
  Vidaio, Vanta, Resi, NOVA, FlameWire, AlphaCore, RedTeam, Hippius.

For research-grade longform, the best outlets are:

- **macrocosmosai.substack.com** — internal R&D blog for SN1/9/13/25.
- **taodaily.io** — daily ecosystem coverage with frequent owner
  interviews.
- **simplytao.ai** — accessible "Simple Guide" series per subnet.
- **podcasts.chainofthought.xyz** — Hash Rate / Chain of Thought podcast
  subnet summaries.
- **backprop.finance/analytics/podcasts** — Bittensor Brief podcast
  episodes (~30 minutes per subnet).
- Subnet-specific Substacks: bitcast.substack.com, synapz.org,
  eli5defi.substack.com, asymmetricjump.substack.com,
  thetaodesk.substack.com, swarm124subnet.substack.com.

---

## Sources used

Top-level discovery: taomarketcap.com, taostats.io, subnetradar.com,
bittensor.co.in, taosubnetguide.com, bittensor.ai, docs.learnbittensor.org.

Per-subnet sources are cited inline. Cross-verified aggregator outlets:

- **taodaily.io** — frequent subnet interviews + recaps.
- **simplytao.ai** — "Simple Guide" per-subnet series.
- **macrocosmosai.substack.com** — research blog for SN1/9/13/25.
- **subnetalpha.ai** — current-state subnet directory.
- **bittensor.co.in** (Bittensor India) — categorized subnet directory.
- **bittensormarketcap.com / dynamictaomarketcap.com** — emissions data.
- **podcasts.chainofthought.xyz** — Hash Rate podcast subnet summaries.
- **backprop.finance/analytics/podcasts** — Bittensor Brief episodes.
- **subnetradar.com** — current health/saturation/market-cap data.

Always cross-check current status on taostats.io before publishing
external content — subnets rebrand, deregister, or change ownership
weekly.
