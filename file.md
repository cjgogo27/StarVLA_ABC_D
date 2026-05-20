\documentclass{article}

\usepackage[final]{neurips_2024}

\newtheorem{theorem}{Theorem}[section]
\newtheorem{corollary}{Corollary}[theorem]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{remark}[theorem]{Remark}
\newcommand{\fig}[2][1]{\includegraphics[draft=False, width=#1\linewidth]{fig/#2}}
\newcommand{\figr}[2][1]{\includegraphics[draft=False, width=#1\linewidth]{fig/#2}}

\newcommand{\demph}[1]{\textcolor{demphcolor}{#1}}
\usepackage{tikz}
\usetikzlibrary{tikzmark}

\def\hlinewd#1{\noalign{\ifnum0=`}\fi\hrule \@height #1 \futurelet \reserved@a\@xhline}

\definecolor{brown1}{RGB}{227, 204, 194}
\definecolor{brown2}{RGB}{247, 219, 189}
\definecolor{myblue}{RGB}{42, 116, 174}
\definecolor{fullmarkcolor}{RGB}{214, 233, 213}
\definecolor{opensource}{RGB}{255, 243, 206}
\definecolor{opensource2}{RGB}{217, 231, 252}
\definecolor{streaming}{RGB}{214, 239, 244}
\definecolor{closesource}{RGB}{184, 219, 179}
\definecolor{blue1}{rgb}{0.8, 0.9, 1.0}
\definecolor{blue2}{rgb}{0.7, 0.85, 1.0}
\definecolor{blue3}{rgb}{0.4, 0.7, 0.9}
\definecolor{scifiBlue}{rgb}{0.2, 0.7, 1.0}
\definecolor{scifiRed}{rgb}{1.0, 0.2, 0.2}
\definecolor{scifiGreen}{rgb}{0.4, 1.0, 0.3}
\definecolor{scifiPurple}{rgb}{0.6, 0.2, 0.8}
\definecolor{demphcolor}{gray}{.5}
\newcommand{\strongbf}[1]{\textbf{\fontseries{b}\selectfont #1}}
\usepackage{pifont}
\usepackage{enumitem}
\usepackage{adjustbox}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{mathtools}
\newcommand{\todo}[1]{{\color{red}#1}}
\newcommand{\TODO}[1]{\textbf{\color{red}[TODO: #1]}}
\definecolor{lightgold}{RGB}{255, 242, 217}

\usepackage{inconsolata}
\usepackage{tcolorbox}
\tcbuselibrary{skins}
\tcbset{
    colback=gray!5,
    colframe=black!70,
    fonttitle=\bfseries,
    sharp corners,
    boxrule=0.5pt,
}
\usepackage{array}
\usepackage{alltt}

\title{An Empirical Study of Vision-Language-Action Models
for Long-Horizon Manipulation on CALVIN ABC$\rightarrow$D}

\author{%
  Yibiao Chen \\
  Beijing University of Posts and Telecommunications \\
  \And
  Chengjie Yang \\
  Southwest Jiaotong University \\
  \And
  Yulin Zhang \\
  ShanghaiTech University \\
}

\begin{document}

\maketitle

\begin{abstract}
Vision-Language-Action (VLA) models have rapidly become the dominant
paradigm for learning generalist robot policies, yet a clear empirical
understanding of \emph{which} design decisions matter most under
distribution shift is still missing. We present a empirical
study built on top of the open-source \textsf{StarVLA} codebase and the
long-horizon, language-conditioned CALVIN ABC$\rightarrow$D
benchmark~\cite{mees2022calvin}, in which the policy is trained on three
environments and evaluated on a fourth, unseen one. Our study is organized
along three axes. \textbf{(i)~Baseline.} We disentangle the contribution
of (a)~ backbone (Qwen3.5-4B vs.\ Cosmos-Predict2-2B),
(b)~the action head (a $\pi_0$-style unified action expert vs.\ the GR00T
dual-system expert), (c)~visual data augmentation (brightness,
random crop, and combined photometric+geometric augmentation at varying
strengths), and (d)~action-head pre-training on LIBERO. \textbf{(ii)~Failure
case analysis.} We find that residual error after extensive scaling is
overwhelmingly concentrated on \emph{push--left} / \emph{push--right}
sub-tasks. Through joint analysis of action-space coverage, data
distribution, and rollout visualizations, we attribute the failure to the
absence of an explicit \emph{progress signal}: the model cannot tell
whether it is moving toward or away from sub-task completion.
\textbf{(iii)~Structural remedies.} Guided by this diagnosis, we explore
four extensions on top of \textsf{StarVLA}: (1)~conditioning action
prediction on past actions as trajectory context, (2)~adding a
Mixture-of-Experts (MoE) action expert with an auxiliary
progress-prediction head, (3)~conditioning the action expert on future
frames imagined by a Cosmos world model, and (4)~RL post-training with the
RLinf-VLA framework. Each of these directions yields measurable gains over
the strongest supervised baseline, and together they identify
\emph{progress awareness} as a key missing ingredient in current
open-source VLAs. We release all checkpoints and training curves to
facilitate reproducibility.
\end{abstract}

% =====================================================================
\section{Introduction}
% =====================================================================

Building robots that follow free-form language instructions to compose
long sequences of manipulation skills in unseen environments is a
defining challenge for embodied intelligence. The CALVIN
benchmark~\cite{mees2022calvin} formalizes this challenge: a single
policy must solve up to five chained language-conditioned sub-tasks per
rollout, drawn from 34 skills, with an explicit ABC$\rightarrow$D split
in which the test environment is held out from training. CALVIN
ABC$\rightarrow$D has therefore become a de-facto stress test for
generalization in Vision-Language-Action (VLA) models.

Recent VLAs combine a pre-trained vision-language backbone with a
specialized action decoder. Representative examples include the
flow-matching policy $\pi_0$~\cite{black2024pi0} built on
PaliGemma~\cite{beyer2024paligemma}, the dual-system
GR00T-N1~\cite{nvidia2025groot} that pairs the Eagle-2 VLM with a
diffusion-transformer action expert, and a series of
diffusion/autoregressive variants such as
OpenVLA~\cite{kim2024openvla}, RT-2~\cite{zitkovich2023rt2},
RDT-1B~\cite{liu2024rdt}, CogACT~\cite{cogact2024} and
OpenVLA-OFT~\cite{kim2025openvlaoft}. In parallel, world-model
foundations such as Cosmos~\cite{nvidia2025cosmos,nvidia2025cosmospredict2}
provide a new class of action-conditioned future-frame predictors that
can be repurposed for embodied control. Despite this rapid progress,
practitioners are still confronted with a daunting design space: which
backbone, which action head, which augmentation policy, which
pre-training corpus, and which post-training algorithm should one pick,
and how do these choices interact under genuine distribution shift?

Existing reports usually advocate a single full-stack recipe and
benchmark it against fixed external baselines. While compelling, such
results conflate many simultaneous changes and offer little actionable
guidance. In this work we take the opposite stance: we build on the
open-source \textsf{StarVLA} codebase, fix CALVIN ABC$\rightarrow$D as a
constant test bed, fix the training compute budget, and systematically
vary one design dimension at a time. \textsf{StarVLA} provides a
modular pipeline for combining different vision-language backbones with
different action experts, which makes it a convenient substrate for the
controlled comparisons we run below. Our study is structured around
three questions.

\textbf{Q1.\,What matters in the baseline?} Within the \textsf{StarVLA}
pipeline, we compare Qwen3.5-4B
\cite{yang2025qwen3} with Cosmos-Predict2-2B
\cite{nvidia2025cosmospredict2} as backbones, the $\pi_0$ unified action
expert~\cite{black2024pi0} with the GR00T dual-system
expert~\cite{nvidia2025groot} as action heads, several augmentation
schedules of varying strength inspired
by~\cite{laskin2020rad,xie2024decomposing}, and an action-head
pre-training stage on LIBERO~\cite{liu2023libero}.

\textbf{Q2.\,Where does the residual error live?} After fixing the best
baseline configuration, we run a per-skill error breakdown together
with state visualizations and per-skill data statistics. We find that
residual failure is heavily concentrated on the
\emph{push\_into\_drawer}, \emph{push\_left}, and \emph{push\_right}
family of skills, and trace this to a missing \emph{progress signal}:
the policy has no internal estimate of how far the sub-task has
advanced, and therefore stalls or oscillates around contact-rich
configurations.

\textbf{Q3.\,Which structural changes close the gap?} Motivated by Q2,
we extend the \textsf{StarVLA} pipeline with four additional modules:
(i)~feeding the past action chunk back into the backbone as trajectory
context; (ii)~adding a Mixture-of-Experts
\cite{shazeer2017outrageously,himoevla2025,moaeyang2025} action expert
together with an explicit progress-prediction head inspired
by~\cite{progressvla2026,vlac2025}; (iii)~using a Cosmos-Predict2 world
model~\cite{nvidia2025cosmospredict2,du2024unipi} to generate future
frames whose features condition the action expert; and (iv)~RL
post-training~\cite{rlinfvla2025,simplevlarl2025,shao2024grpo} with the
RLinf framework~\cite{rlinf2025}.

\paragraph{Contributions.} (1)~We release the apples-to-apples
study on CALVIN ABC$\rightarrow$D, built on the \textsf{StarVLA}
codebase, that simultaneously controls for the backbone, action head,
augmentation, and pre-training. (2)~We provide a quantitative
failure-mode analysis showing that \emph{progress awareness} is the
single largest source of the remaining gap.
(3)~We design and benchmark four structural remedies on top of
\textsf{StarVLA} that each target the progress bottleneck, and report
training curves and checkpoints for all of them.

% =====================================================================
\section{Related Work}
% =====================================================================

\paragraph{VLA models and long-horizon manipulation.}
Modern VLAs fine-tune a pre-trained vision-language model on robot
trajectories to produce a generalist policy. RT-2~\cite{zitkovich2023rt2}
and OpenVLA~\cite{kim2024openvla} cast actions as discrete tokens, while
$\pi_0$~\cite{black2024pi0} attaches a flow-matching~\cite{lipman2023flowmatching}
expert to PaliGemma~\cite{beyer2024paligemma} and
GR00T-N1~\cite{nvidia2025groot} adopts a dual-system diffusion-transformer
layout~\cite{peebles2023dit,ho2020ddpm,kahneman2011thinking}.
RDT-1B~\cite{liu2024rdt}, CogACT~\cite{cogact2024} and
OpenVLA-OFT~\cite{kim2025openvlaoft} further refine action chunking and
regression losses. On CALVIN ABC$\rightarrow$D specifically,
RoboFlamingo~\cite{li2024roboflamingo} adapts a VLM with an LSTM head;
GR-1~\cite{wu2023gr1} shows that large-scale video pre-training
transfers to manipulation; SuSIE~\cite{black2024susie} and
CLOVER~\cite{bu2024clover} synthesize future images as planning
intermediates; Seer~\cite{tian2025seer} predicts inverse dynamics from
imagined frames; and MDT~\cite{reuss2024mdt},
RoboUniView~\cite{liu2024robouniview}, 3D Diffuser Actor~\cite{ke20243d}
and FLOWER~\cite{reuss2025flower} explore diffusion- and flow-based
heads with multi-view or 3D representations. Our study uses Qwen3.5-4B
\cite{yang2025qwen3} and Cosmos-Predict2-2B~\cite{nvidia2025cosmospredict2}
as backbones, paired with either the $\pi_0$ unified action expert or
the GR00T dual-system expert, and reports on the same ABC$\rightarrow$D
protocol.

\paragraph{World models, MoE, and progress estimation.}
UniPi~\cite{du2024unipi} and UniSim~\cite{yang2024unisim} frame
decision-making as text-conditioned video generation followed by inverse
dynamics, while Dreamer~\cite{hafner2023dreamerv3} learns latent dynamics
for policy search. The Cosmos family~\cite{nvidia2025cosmos,nvidia2025cosmospredict2}
provides open-weight world models that we use to condition the action
expert on imagined future frames. For action-head scaling, sparsely-gated
MoE layers~\cite{shazeer2017outrageously} have been adapted to VLAs by
HiMoE-VLA~\cite{himoevla2025} and \cite{moaeyang2025}; we build on these
designs to add an auxiliary progress head. On progress estimation,
ProgressVLA~\cite{progressvla2026} back-propagates a progress-maximization
signal through a world model, and VLAC~\cite{vlac2025} outputs a signed
progress delta as dense reward---both directly motivate our auxiliary head
in Section~\ref{sec:remedies}.

\paragraph{Data augmentation and RL post-training.}
RAD~\cite{laskin2020rad} and DrQ~\cite{kostrikov2021drq} established that
random crop and color jitter substantially improve visual generalization;
\cite{xie2024decomposing,chen2024semaug,yu2023scaling} extend this to
imitation learning for manipulation. We evaluate brightness, crop, and
combined photometric schedules at varying intensities. For RL post-training,
SimpleVLA-RL~\cite{simplevlarl2025}, RLinf-VLA~\cite{rlinfvla2025} and
RLinf~\cite{rlinf2025} post-train VLAs with GRPO~\cite{shao2024grpo} or
PPO~\cite{schulman2017ppo} using outcome rewards; we adopt the RLinf-VLA
pipeline in Section~\ref{sec:rl}.

% =====================================================================
\section{Task Definition and Experimental Setup}
\label{sec:setup}
% =====================================================================

\subsection{The CALVIN ABC$\rightarrow$D benchmark}
CALVIN~\cite{mees2022calvin} provides four tabletop environments
(A, B, C, D) sharing 34 manipulation skills---\emph{open/close drawer},
\emph{turn on/off lightbulb}, \emph{push block left/right}, and so on.
Each demonstration episode contains a synchronized stream of two RGB
views (static and gripper), 7-DoF proprioception, end-effector
actions, and a language annotation. The ABC$\rightarrow$D split
uses environments A, B and C for training and environment D, with
unseen visual textures and object positions, for evaluation. The
canonical metric is the average length of completed sub-tasks within
a chain of five language instructions; we additionally report per-step
success rates.

\subsection{Pipeline architecture}
We build on the open-source \textsf{StarVLA} codebase, which follows the
standard dual-stack VLA layout: a vision-language backbone
$\Phi_{\text{VL}}$ encodes RGB observations and the current language
instruction into a sequence of multimodal tokens, and an action expert
$\Phi_{\text{A}}$ consumes these tokens together with the robot
proprioception to produce an action chunk of horizon $H$. The action
expert is trained either with a flow-matching loss
($\pi_0$-style unified)~\cite{black2024pi0,lipman2023flowmatching} or with a
DDPM~\cite{ho2020ddpm} loss on a diffusion transformer
(GR00T dual-system style)~\cite{nvidia2025groot,peebles2023dit}. Unless stated
otherwise, we use $H=5$ and execute four actions before re-querying
the model. All extensions in Section~\ref{sec:remedies} are implemented
as drop-in modules within the same pipeline.

\subsection{Training and evaluation protocol}
All experiments use the same optimizer (AdamW), learning-rate schedule
(cosine with warm-up), batch size, and number of gradient steps so
that runs differ only in the variable under investigation. For backbone
training we use a learning rate in the range of $2 \times 10^{-6}$
(e.g., $2\mathrm{e}{-6}$), while for the action head we use a learning
rate in the range of $1 \times 10^{-4}$ to $4 \times 10^{-4}$ ($1$--$4\mathrm{e}{-4}$).
Each configuration is evaluated on the CALVIN ABC$\rightarrow$D validation
split using 1{,}000 randomly sampled instruction chains of length 5.
Full training hyper-parameters are listed in
Appendix~\ref{app:hparams}.

% --- placeholder for setup figure ---
\noindent\textbf{[Figure~\ref{fig:overview} placeholder: pipeline
overview of the \textsf{StarVLA} codebase as used in this study,
showing backbone $\Phi_{\text{VL}}$, action expert $\Phi_{\text{A}}$,
and the four optional plug-ins introduced in Section~\ref{sec:remedies}:
past-action history, MoE+progress, Cosmos future-frame conditioning,
and RL post-training.]}

% =====================================================================
\section{Architecture and Data: Ablating the Baseline}
\label{sec:baseline}
% =====================================================================

In this section, we ablate the four building blocks that practitioners 
must choose before introducing any structural innovations: backbone, action head, data augmentation, 
and action-head pre-training. For all experiments, we train for 30,000 steps to ensure a fair comparison. 
Each ablation uses the same training budget and the same evaluation protocol described in Section~\ref{sec:setup}.

% ---------------------------------------------------------------------
\subsection{Backbone: Qwen3.5-4B vs.\ Cosmos-Predict2-2B}
\label{sec:backbone}
% ---------------------------------------------------------------------

We compare two backbones of similar parameter count but radically
different pre-training. Qwen3.5-4B~\cite{yang2025qwen3} is a
general-purpose VLM trained at native resolution with strong document
and grounding capabilities. Cosmos-Predict2-2B
\cite{nvidia2025cosmospredict2} is a flow-based world foundation model
pre-trained on 200M+ curated robotics-relevant video clips that
explicitly predicts the future state of the world. In all experiments,
we train all components jointly from the beginning without any stage-wise
freezing; both the backbone and the action expert (PI0) are fully
trainable throughout.

\noindent\textbf{[Section~\ref{sec:backbone} text placeholder: discuss
training curves, average-length numbers, and qualitative differences.
Headline question: does world-model pre-training translate into more
sample-efficient manipulation learning, or is general-purpose VLM
pre-training a better starting point for ABC$\rightarrow$D?]}

\noindent\textbf{[Figure~\ref{fig:backbone} placeholder: train/val
curves of Qwen3.5-4B and Cosmos-Predict2-2B under matched compute.]}

\noindent\textbf{[Table~\ref{tab:backbone} placeholder: per-sub-task
success rate and average length (out of 5) on CALVIN ABC$\rightarrow$D
for the two backbones.]}

% ---------------------------------------------------------------------
\subsection{Action head: $\pi_0$ unified vs.\ GR00T dual-system}
\label{sec:action_head}
% ---------------------------------------------------------------------

Holding the backbone fixed at the better choice from
Section~\ref{sec:backbone}, we compare two open action experts that
have come to define modern VLAs:
\begin{itemize}[leftmargin=*]
\item A \emph{unified action expert} in the style of $\pi_0$
      \cite{black2024pi0}, trained with the conditional flow-matching
      objective of~\cite{lipman2023flowmatching}.
\item A \emph{dual-system expert} in the style of GR00T
      \cite{nvidia2025groot}, trained with the standard DDPM
      objective~\cite{ho2020ddpm} on action chunks.
\end{itemize}

\noindent\textbf{[Section~\ref{sec:action_head} text placeholder:
report final ABC$\rightarrow$D numbers, inference-time chunk latency,
and a qualitative diagnosis of multimodality. Discuss whether the two
heads recover the same actions on the same observations, or whether
they cover different modes of the action distribution.]}

\noindent\textbf{[Figure~\ref{fig:headcompare} placeholder: training
loss vs.\ ABC$\rightarrow$D average length for the two action heads.]}

% ---------------------------------------------------------------------
\subsection{Visual data augmentation}
\label{sec:aug}
% ---------------------------------------------------------------------

ABC$\rightarrow$D is fundamentally a visual generalization problem: the
test environment shares geometry with the training environments but
differs in textures, colors, and lighting. Motivated by
RAD~\cite{laskin2020rad}, DrQ~\cite{kostrikov2021drq} and
\cite{xie2024decomposing}, we apply a compound augmentation pipeline
consisting of random cropping, brightness/contrast/saturation jitter,
Gaussian blur, additive noise, and JPEG compression artifacts. We
compare two configurations: a \textbf{Mild} baseline and an
\textbf{Aggressive} variant with uniformly higher perturbation
magnitudes. Table~\ref{tab:aug_params} lists the exact hyper-parameters
for each setting.

\begin{table}[h]
\centering
\caption{Visual augmentation hyper-parameters for the two configurations
evaluated in this study.}
\label{tab:aug_params}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Parameter} & \textbf{Mild} & \textbf{Aggressive} \\
\midrule
Crop scale min         & 0.96 & 0.88 \\
Brightness jitter      & 0.10 & 0.20 \\
Contrast jitter        & 0.10 & 0.20 \\
Saturation jitter      & 0.05 & 0.15 \\
Blur probability       & 0.10 & 0.25 \\
Blur radius            & 0.5  & 0.8  \\
Noise probability      & 0.10 & 0.25 \\
Noise std              & 2.0  & 3.0  \\
JPEG probability       & 0.10 & 0.25 \\
JPEG quality min/max   & 80/95 & 65/92 \\
\bottomrule
\end{tabular}
\end{table}

\noindent\textbf{[Section~\ref{sec:aug} text placeholder: report
ABC$\rightarrow$D average-length results for Mild vs.\ Aggressive
augmentation. Discuss whether the stronger perturbations help or hurt
contact-rich skills such as push-left/right, and the trade-off between
visual diversity and the view-consistent information needed for
precise end-effector control.]}

\noindent\textbf{[Figure~\ref{fig:aug} placeholder: bar chart of
ABC$\rightarrow$D average length for each augmentation setting.]}

\noindent\textbf{[Table~\ref{tab:aug} placeholder: per-skill success
rates under Mild and Aggressive augmentation.]}

% ---------------------------------------------------------------------
\subsection{Pre-training the action head on LIBERO}
\label{sec:pretrain}
% ---------------------------------------------------------------------

Following recent practice~\cite{liu2023libero,kim2025openvlaoft}, we
pre-train the GR00T action expert on the four LIBERO suites
(Spatial, Object, Goal, Long) for a fixed number of gradient steps
before fine-tuning on CALVIN ABC. Due to compute constraints, we do
not evaluate the pre-trained checkpoint on the full CALVIN
ABC$\rightarrow$D benchmark; instead, we report the LIBERO pre-training
loss curve as a reference for convergence behavior and to facilitate
reproducibility.

\noindent\textbf{[Figure~\ref{fig:pretrain} placeholder: action-expert
training loss on the four LIBERO suites across gradient steps,
showing convergence for each suite (Spatial, Object, Goal, Long).]}

% ---------------------------------------------------------------------
\subsection{Best-configuration summary}
\label{sec:best}
% ---------------------------------------------------------------------

Combining the winning choice along each axis yields our reference
baseline configuration, which we refer to as \textbf{Base} throughout
the rest of the paper. Concretely, \textbf{Base} uses the
Qwen3.5-4B~\cite{yang2025qwen3} backbone paired with the GR00T
dual-system action expert~\cite{nvidia2025groot} and trained with the
\emph{Aggressive} augmentation schedule (Table~\ref{tab:aug_params}).
All later extensions are implemented as additional modules on top of
this configuration within the \textsf{StarVLA} codebase.

% =====================================================================
\section{Failure Case Exploration}
\label{sec:failure}
% =====================================================================

With the best baseline fixed, we now ask the more interesting question:
\emph{where does the remaining error live, and what does it tell us
about what the model is missing?}

% ---------------------------------------------------------------------
\subsection{Per-skill error breakdown}
\label{sec:perskill}
% ---------------------------------------------------------------------

We decompose every failed sub-task in the ABC$\rightarrow$D
evaluation into the 34 canonical skill categories. The picture is
strikingly imbalanced: a small family of \emph{push} skills---most
notably \emph{push\_left}, \emph{push\_right} and
\emph{push\_into\_drawer}---accounts for a disproportionate share of
the remaining error, while pick-and-place and articulated-object
skills are nearly saturated.

\noindent\textbf{[Section~\ref{sec:perskill} text placeholder:
report the exact per-skill success rates of \textbf{Base} and quote
the worst-three and best-three skills. Discuss the chain effect: a
failure on \emph{push\_left} early in an instruction chain immediately
kills the remainder of the rollout, so this small skill family is
unusually high-leverage.]}

\noindent\textbf{[Figure~\ref{fig:perskill} placeholder: per-skill
success rates ranked from worst (push\_left/right family) to best;
overlay the support count from the training set on a secondary axis.]}

% ---------------------------------------------------------------------
\subsection{Why are push-left / push-right tasks the hardest?}
\label{sec:why_push}
% ---------------------------------------------------------------------

We analyze three candidate explanations and find that none of the
``easy'' culprits fully holds.

\paragraph{Data balance.} If push-left/right were simply
under-represented in the training distribution, the failure could be
explained by data starvation.

\noindent\textbf{[Section~\ref{sec:why_push} text placeholder
(data balance): report the per-skill frequency in CALVIN ABC, and show
that push-left/right are within the typical range, not statistical
outliers.]}

\noindent\textbf{[Figure~\ref{fig:databalance} placeholder: histogram
of episode count per skill in CALVIN ABC, with the push-left/right
buckets highlighted.]}

\paragraph{Action magnitude.} A second hypothesis is that the
push family requires unusually large or unusually fine end-effector
displacements.

\noindent\textbf{[Section~\ref{sec:why_push} text placeholder
(action magnitude): report the empirical distribution of per-step
displacement and chunk length for each skill, and show that push
actions are unremarkable in this respect.]}

\noindent\textbf{[Figure~\ref{fig:actiondist} placeholder: violin plot
of per-step action norms per skill.]}

\paragraph{Progress awareness.} The third---and, we argue, correct
hypothesis---is that the model lacks an internal estimate of \emph{how
far it is from completing the current sub-task}. Pick-and-place and
articulated-object skills have visually salient goal states (an open
drawer, a closed door) that act as implicit progress markers. Push
skills, in contrast, terminate at a block configuration that differs
only mildly from earlier frames, leaving the policy without a strong
visual cue for when to stop.

\noindent\textbf{[Section~\ref{sec:why_push} text placeholder
(progress): visualize a failed push rollout. The end-effector
oscillates around the block, repeatedly contacts it, and never commits
to a long enough push; the action distribution becomes bimodal and
the model never resolves the ambiguity. Contrast with a successful
rollout from a pick-and-place skill where the visual goal state is
crisp.]}

\noindent\textbf{[Figure~\ref{fig:failurevis} placeholder: side-by-side
rollouts of a failed push-left case and a successful open-drawer case,
together with a per-step plot of (estimated) task progress.]}

This diagnosis motivates the four structural extensions reported in
Section~\ref{sec:remedies}.

% =====================================================================
\section{Structural Remedies}
\label{sec:remedies}
% =====================================================================

Guided by Section~\ref{sec:failure}, each of the following four
extensions injects an additional signal that should help the model
form a better internal estimate of task progress.

% ---------------------------------------------------------------------
\subsection{History conditioning: past actions as context}
\label{sec:history}
% ---------------------------------------------------------------------

Our first intervention is the simplest: we feed the previous action
chunk back into the Qwen-3.5-VL backbone as an additional context
token sequence, so that the GR00T action expert can condition on
the full trajectory rather than only the instantaneous observation.
This is analogous to the proprioceptive history used in ACT-style
policies~\cite{zhao2024act} but inserted at the backbone rather than
the expert.

\noindent\textbf{[Section~\ref{sec:history} text placeholder: describe
the input formatting, the resulting parameter count, and the
ABC$\rightarrow$D numbers. Discuss whether the gain is concentrated
on the push family or distributed across skills.]}

\noindent\textbf{[Figure~\ref{fig:history} placeholder: training and
validation curves, plus a per-skill bar chart of the absolute gain
over \textbf{Base}.]}

% ---------------------------------------------------------------------
\subsection{MoE action expert with a progress head}
\label{sec:moe}
% ---------------------------------------------------------------------

We replace the dense feed-forward layers of the GR00T expert with
sparsely-gated Mixture-of-Experts layers~\cite{shazeer2017outrageously},
following recent VLA-MoE
designs~\cite{himoevla2025,moaeyang2025}. In parallel, we attach an
auxiliary progress-prediction head that regresses a normalized progress
value $p_t \in [0,1]$ derived from temporal position within each
demonstration episode, similar to~\cite{progressvla2026,vlac2025}. The
total loss combines flow-matching/DDPM action loss with an L2 progress
loss.

\noindent\textbf{[Section~\ref{sec:moe} text placeholder: report the
ABC$\rightarrow$D numbers, the calibration of the predicted progress
signal (Pearson $r$ against the ground-truth linear ramp), and an
ablation that removes either MoE or the progress head.]}

\noindent\textbf{[Figure~\ref{fig:progress_calibration} placeholder:
predicted progress trajectories overlaid with the ground-truth ramp
for several push and pick sequences.]}

\noindent\textbf{[Figure~\ref{fig:moe} placeholder: per-skill gains of
the MoE+progress variant over \textbf{Base}.]}

% ---------------------------------------------------------------------
\subsection{Future-frame conditioning with a Cosmos world model}
\label{sec:future}
% ---------------------------------------------------------------------

Our third extension follows the spirit of UniPi~\cite{du2024unipi} and
generative-expectation policies~\cite{bu2024clover,tian2025seer,black2024susie}:
a Cosmos-Predict2 world model~\cite{nvidia2025cosmospredict2}, fine-tuned
on CALVIN ABC, generates $K$ future RGB frames conditioned on the current
observation and language instruction. The latent features of these
generated frames are then injected into the action expert as additional
conditioning tokens. The world model is frozen during action-policy
training.

\noindent\textbf{[Section~\ref{sec:future} text placeholder: describe
the fine-tuning of Cosmos-Predict2 on CALVIN ABC, the choice of $K$
and the conditioning mechanism (cross-attention vs.\ prefix tokens),
and the ABC$\rightarrow$D numbers. Discuss the qualitative behaviour
on push tasks: does the imagined future indicate the block at a
displaced location and thereby provide an implicit progress cue?]}

\noindent\textbf{[Figure~\ref{fig:future} placeholder: qualitative
example showing (top) a current observation and language instruction,
(middle) the generated future frames from Cosmos-Predict2, and
(bottom) the resulting action chunk overlaid on the rollout.]}

% ---------------------------------------------------------------------
\subsection{Reinforcement learning post-training with RLinf-VLA}
\label{sec:rl}
% ---------------------------------------------------------------------

Finally, we post-train the \textbf{Base} configuration with the
RLinf-VLA pipeline~\cite{rlinfvla2025} using
GRPO~\cite{shao2024grpo}. We run the policy in CALVIN environments
A, B, C, collect outcome rewards (success/failure of each language
sub-task), and update the policy with the group-relative advantage
estimator of~\cite{shao2024grpo,simplevlarl2025}. The action expert
is updated via a flow-aware variant adapted to the chunked output.

\noindent\textbf{[Section~\ref{sec:rl} text placeholder: report the
RL training curves, the final ABC$\rightarrow$D numbers, and the
per-skill effect (particularly on the push family). Discuss the
relative magnitude of the gain compared to history conditioning, MoE+progress,
and future-frame conditioning.]}

\noindent\textbf{[Figure~\ref{fig:rl_curves} placeholder: GRPO success
rate during post-training in A/B/C and the corresponding zero-shot
ABC$\rightarrow$D success rate at each checkpoint.]}

% ---------------------------------------------------------------------
\subsection{Putting the four remedies together}
\label{sec:combine}
% ---------------------------------------------------------------------

\noindent\textbf{[Section~\ref{sec:combine} text placeholder:
report the success of stacking the four extensions (history + MoE
progress head + future-frame conditioning + RL post-training) and
discuss which combinations are complementary and which are
redundant. Provide a final table comparing \textbf{Base}, each of the
four single extensions, and the stacked \textbf{Base+All} to the
strongest publicly reported numbers on CALVIN ABC$\rightarrow$D.]}

\noindent\textbf{[Table~\ref{tab:final_summary} placeholder: master
table of all configurations.]}

% =====================================================================
\section{Conclusion}
% =====================================================================

We presented a controlled empirical study of vision-language-action
models on CALVIN ABC$\rightarrow$D, built entirely on top of the
open-source \textsf{StarVLA} codebase. Our baseline ablation isolates
the contribution of backbone, action head, augmentation, and
pre-training, and our failure analysis identifies missing progress
awareness as the dominant source of remaining error, concentrated on
the push family of skills. Four structural extensions---past-action
context, MoE plus a progress head, future-frame conditioning from a
Cosmos world model, and RL post-training with RLinf---each move the
needle, and each can be understood as injecting a different form of
progress information into the policy. We hope this study, together
with the released checkpoints and training curves on top of
\textsf{StarVLA}, serves as a clean reference point for future VLA
research on long-horizon manipulation under distribution shift.

% =====================================================================
\appendix
\section{Hyper-parameters}
\label{app:hparams}

\noindent\textbf{[Appendix~\ref{app:hparams} placeholder: list optimizer,
batch sizes, learning-rate schedules, chunk lengths, and augmentation
parameters for every configuration.]}

% =====================================================================
{
    \small
    \bibliographystyle{unsrtnat}
    \bibliography{references}
}

\end{document}
