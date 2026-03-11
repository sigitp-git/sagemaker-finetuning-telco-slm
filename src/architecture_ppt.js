/**
 * Generate architecture-diagram.pptx — visual architecture and flow diagrams
 * for the 3GPP RCA fine-tuning benchmark.
 *
 * Usage:
 *   node src/architecture_ppt.js
 *   Output: reports/architecture-diagram.pptx
 */

const PptxGenJS = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pptx.title = "3GPP RCA Benchmark — Architecture";
pptx.author = "SageMaker Fine-Tuning Benchmark";

// ── Palette ──────────────────────────────────────────────────────────────────
const C = {
  DARK: "0A1628", WHITE: "FFFFFF", BLUE: "0066CC", LIGHT_BLUE: "E8F4FD",
  ORANGE: "FF9900", LIGHT_ORANGE: "FFF4E5", GREEN: "1B8A2D", LIGHT_GREEN: "E6F5E8",
  PURPLE: "6B21A8", LIGHT_PURPLE: "F3E8FF", GRAY: "64748B", LIGHT_GRAY: "F1F5F9",
  RED: "DC2626", TEAL: "0D9488", LIGHT_TEAL: "E6FAF8", YELLOW: "CA8A04",
};

const FONT = "Arial";

// ── Helper functions ─────────────────────────────────────────────────────────
function box(slide, x, y, w, h, fill, borderColor, text, opts = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h, fill: { color: fill },
    line: { color: borderColor, width: 1.5 },
    rectRadius: 0.08,
    shadow: { type: "outer", blur: 3, offset: 1, color: "00000020" },
  });
  if (text) {
    slide.addText(text, {
      x, y, w, h, fontSize: opts.fontSize || 10, fontFace: FONT,
      color: opts.color || C.DARK, align: "center", valign: "middle",
      bold: opts.bold || false, lineSpacingMultiple: opts.lineSpacing || 1.1,
    });
  }
}

function arrow(slide, x1, y1, x2, y2, color = C.GRAY, opts = {}) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width: opts.width || 1.5, dashType: opts.dash || "solid",
            endArrowType: "triangle" },
  });
}

function label(slide, x, y, w, text, opts = {}) {
  slide.addText(text, {
    x, y, w, h: opts.h || 0.3, fontSize: opts.fontSize || 8, fontFace: FONT,
    color: opts.color || C.GRAY, align: opts.align || "center",
    italic: opts.italic || false, bold: opts.bold || false,
  });
}

function sectionTitle(slide, text) {
  slide.addText(text, {
    x: 0.4, y: 0.2, w: 12.5, h: 0.55,
    fontSize: 22, bold: true, color: C.DARK, fontFace: FONT,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 0.4, y: 0.75, w: 12.5, h: 0,
    line: { color: C.BLUE, width: 2 },
  });
}

function swimlaneHeader(slide, x, y, w, h, text, color) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h, fill: { color }, rectRadius: 0.06,
  });
  slide.addText(text, {
    x, y, w, h, fontSize: 10, bold: true, color: C.WHITE,
    fontFace: FONT, align: "center", valign: "middle",
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ══════════════════════════════════════════════════════════════════════════════
const s1 = pptx.addSlide();
s1.background = { color: C.DARK };
s1.addText("3GPP RCA Benchmark", {
  x: 0.8, y: 1.0, w: 11.7, h: 1.2,
  fontSize: 40, bold: true, color: C.WHITE, align: "center", fontFace: FONT,
});
s1.addText("Architecture and Data Flow", {
  x: 0.8, y: 2.3, w: 11.7, h: 0.7,
  fontSize: 22, color: C.LIGHT_BLUE, align: "center", fontFace: FONT,
});
s1.addText(
  "Fine-tuning 14B SLMs on SageMaker AI for 5G Root Cause Analysis\n" +
  "Hugging Face  ·  Amazon SageMaker  ·  Amazon Bedrock  ·  Amazon S3",
  {
    x: 1.5, y: 3.5, w: 10.3, h: 0.9,
    fontSize: 13, color: "8899BB", align: "center", fontFace: FONT,
    lineSpacingMultiple: 1.4,
  }
);
// Three model icons at bottom
const models = [
  { name: "Mistral-Nemo\n14B", color: C.ORANGE },
  { name: "Qwen3\n14B", color: C.BLUE },
  { name: "Gemma 3\n12B", color: C.GREEN },
];
models.forEach((m, i) => {
  const bx = 3.5 + i * 2.3;
  s1.addShape(pptx.ShapeType.roundRect, {
    x: bx, y: 5.0, w: 1.8, h: 0.9,
    fill: { color: m.color }, rectRadius: 0.08,
  });
  s1.addText(m.name, {
    x: bx, y: 5.0, w: 1.8, h: 0.9,
    fontSize: 11, bold: true, color: C.WHITE, align: "center",
    valign: "middle", fontFace: FONT,
  });
});
s1.addText(new Date().toISOString().slice(0, 10), {
  x: 0.8, y: 6.5, w: 11.7, h: 0.4,
  fontSize: 10, color: "556688", align: "center", fontFace: FONT,
});

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — End-to-End Pipeline Overview (high-level flow)
// ══════════════════════════════════════════════════════════════════════════════
const s2 = pptx.addSlide();
sectionTitle(s2, "End-to-End Pipeline Overview");

// 6 pipeline stages as a horizontal flow
const stages = [
  { label: "1. Data\nGeneration", sub: "Amazon Bedrock\nClaude Opus", fill: C.LIGHT_PURPLE, border: C.PURPLE, icon: "🧠" },
  { label: "2. Data\nStorage", sub: "Amazon S3\ntrain.jsonl / test.jsonl", fill: C.LIGHT_GREEN, border: C.GREEN, icon: "📦" },
  { label: "3. Fine-Tune\nSLMs", sub: "SageMaker AI\nTraining Jobs", fill: C.LIGHT_ORANGE, border: C.ORANGE, icon: "⚡" },
  { label: "4. Evaluate\nFrontier", sub: "Amazon Bedrock\nConverse API", fill: C.LIGHT_PURPLE, border: C.PURPLE, icon: "🔍" },
  { label: "5. Inference\nSLMs", sub: "SageMaker AI\nTraining Jobs", fill: C.LIGHT_ORANGE, border: C.ORANGE, icon: "🎯" },
  { label: "6. Score &\nReport", sub: "Local Python\nscikit-learn", fill: C.LIGHT_BLUE, border: C.BLUE, icon: "📊" },
];

const stageW = 1.7, stageH = 1.5, stageY = 1.3, startX = 0.5, gap = 0.35;
stages.forEach((st, i) => {
  const sx = startX + i * (stageW + gap);
  box(s2, sx, stageY, stageW, stageH, st.fill, st.border, "", {});
  s2.addText(st.label, {
    x: sx, y: stageY + 0.15, w: stageW, h: 0.7,
    fontSize: 11, bold: true, color: C.DARK, fontFace: FONT, align: "center",
    valign: "middle", lineSpacingMultiple: 1.1,
  });
  s2.addText(st.sub, {
    x: sx, y: stageY + 0.85, w: stageW, h: 0.55,
    fontSize: 8, color: C.GRAY, fontFace: FONT, align: "center",
    valign: "middle", lineSpacingMultiple: 1.1,
  });
  if (i < stages.length - 1) {
    arrow(s2, sx + stageW + 0.02, stageY + stageH / 2, sx + stageW + gap - 0.02, stageY + stageH / 2, C.GRAY, { width: 2 });
  }
});

// Bottom: SDK/Library layer
const sdkY = 3.3;
s2.addShape(pptx.ShapeType.roundRect, {
  x: 0.5, y: sdkY, w: 12.3, h: 0.7,
  fill: { color: C.LIGHT_GRAY }, line: { color: C.GRAY, width: 1 }, rectRadius: 0.06,
});
s2.addText("SDKs & Libraries:  SageMaker Python SDK  ·  boto3  ·  transformers  ·  peft  ·  trl  ·  bitsandbytes  ·  accelerate  ·  datasets  ·  scikit-learn", {
  x: 0.5, y: sdkY, w: 12.3, h: 0.7,
  fontSize: 10, color: C.DARK, fontFace: FONT, align: "center", valign: "middle",
});

// Models row
const modelY = 4.4;
label(s2, 0.5, modelY - 0.05, 12.3, "Models Fine-Tuned (QLoRA 4-bit)", { fontSize: 11, bold: true, color: C.DARK });
const modelCards = [
  { name: "Mistral-Nemo-Base-2407", org: "mistralai", gpu: "1× A10G (ml.g5.2xlarge)", color: C.ORANGE },
  { name: "Qwen3-14B", org: "Qwen", gpu: "4× A10G (ml.g5.12xlarge)", color: C.BLUE },
  { name: "Gemma 3 12B", org: "google", gpu: "1× A10G (ml.g5.2xlarge)", color: C.GREEN },
];
modelCards.forEach((mc, i) => {
  const mx = 0.5 + i * 4.2;
  box(s2, mx, modelY + 0.35, 3.8, 0.9, C.WHITE, mc.color, "", {});
  s2.addText(mc.name, {
    x: mx, y: modelY + 0.38, w: 3.8, h: 0.4,
    fontSize: 11, bold: true, color: mc.color, fontFace: FONT, align: "center",
  });
  s2.addText(`${mc.org}  ·  ${mc.gpu}`, {
    x: mx, y: modelY + 0.75, w: 3.8, h: 0.35,
    fontSize: 8, color: C.GRAY, fontFace: FONT, align: "center",
  });
});

// Frontier models row
const frontierY = 5.8;
label(s2, 0.5, frontierY - 0.05, 12.3, "Frontier Models (Amazon Bedrock)", { fontSize: 11, bold: true, color: C.DARK });
const frontierCards = [
  { name: "Claude Opus 4.6", id: "us.anthropic.claude-opus-4-6-v1", color: C.PURPLE },
  { name: "Amazon Nova Pro", id: "amazon.nova-pro-v1:0", color: C.TEAL },
];
frontierCards.forEach((fc, i) => {
  const fx = 2.5 + i * 4.5;
  box(s2, fx, frontierY + 0.35, 3.8, 0.7, C.WHITE, fc.color, "", {});
  s2.addText(fc.name, {
    x: fx, y: frontierY + 0.35, w: 3.8, h: 0.35,
    fontSize: 11, bold: true, color: fc.color, fontFace: FONT, align: "center",
  });
  s2.addText(fc.id, {
    x: fx, y: frontierY + 0.65, w: 3.8, h: 0.3,
    fontSize: 7, color: C.GRAY, fontFace: FONT, align: "center",
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — S3 Bucket Architecture (tree diagram)
// ══════════════════════════════════════════════════════════════════════════════
const s3slide = pptx.addSlide();
sectionTitle(s3slide, "Amazon S3 Bucket Architecture");

// Bucket root
box(s3slide, 4.5, 1.1, 4.3, 0.6, C.LIGHT_GREEN, C.GREEN,
  "s3://your-telco-llm-bucket/", { fontSize: 12, bold: true, color: C.GREEN });

// Four main folders
const folders = [
  { name: "data/", desc: "Training & test JSONL", x: 0.3, color: C.BLUE,
    files: ["train.jsonl (1,300 examples)", "test.jsonl (992 examples)"],
    writer: "Manual upload" },
  { name: "output/", desc: "LoRA adapter weights", x: 3.6, color: C.ORANGE,
    files: ["mistral-nemo-base-2407/\n  output.tar.gz → adapter/",
            "qwen3-14b/\n  output.tar.gz → adapter/",
            "gemma-3-12b-pt/\n  output.tar.gz → adapter/"],
    writer: "SageMaker Training Job" },
  { name: "inference-output/", desc: "SLM predictions", x: 7.0, color: C.PURPLE,
    files: ["mistral-nemo-base-2407/\n  preds_*_slm.jsonl",
            "qwen3-14b/\n  preds_*_slm.jsonl",
            "gemma-3-12b-pt/\n  preds_*_slm.jsonl"],
    writer: "SageMaker Inference Job" },
  { name: "results/", desc: "Evaluation metrics", x: 10.5, color: C.TEAL,
    files: ["results.json", "results_final.json"],
    writer: "Manual upload after scoring" },
];

folders.forEach((f) => {
  const fy = 2.2;
  // Arrow from bucket to folder
  arrow(s3slide, 6.65, 1.7, f.x + 1.3, fy, C.GRAY, { width: 1 });

  // Folder header
  box(s3slide, f.x, fy, 2.6, 0.5, f.color, f.color, f.name, {
    fontSize: 11, bold: true, color: C.WHITE });

  // Description
  label(s3slide, f.x, fy + 0.5, 2.6, f.desc, { fontSize: 8, color: C.GRAY });

  // Files
  f.files.forEach((file, i) => {
    const fileY = fy + 0.85 + i * 0.55;
    s3slide.addShape(pptx.ShapeType.roundRect, {
      x: f.x + 0.1, y: fileY, w: 2.4, h: 0.48,
      fill: { color: C.LIGHT_GRAY }, line: { color: "D0D7E3", width: 0.5 },
      rectRadius: 0.04,
    });
    s3slide.addText(file, {
      x: f.x + 0.2, y: fileY, w: 2.2, h: 0.48,
      fontSize: 7, color: C.DARK, fontFace: FONT, align: "left", valign: "middle",
      lineSpacingMultiple: 1.0,
    });
  });

  // Writer label
  const writerY = fy + 0.85 + f.files.length * 0.55 + 0.1;
  label(s3slide, f.x, writerY, 2.6, `← ${f.writer}`, {
    fontSize: 7, italic: true, color: f.color });
});

// Legend at bottom
const legY = 6.3;
s3slide.addShape(pptx.ShapeType.roundRect, {
  x: 0.3, y: legY, w: 12.7, h: 0.7,
  fill: { color: C.LIGHT_BLUE }, line: { color: C.BLUE, width: 0.5 }, rectRadius: 0.06,
});
s3slide.addText(
  "Data flow:  Manual upload → data/  |  submit_training.py → output/  |  submit_inference.py → inference-output/  |  evaluate.py → results/",
  { x: 0.3, y: legY, w: 12.7, h: 0.7, fontSize: 9, color: C.DARK, fontFace: FONT, align: "center", valign: "middle" }
);

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — SageMaker Training Job Flow (detailed)
// ══════════════════════════════════════════════════════════════════════════════
const s4 = pptx.addSlide();
sectionTitle(s4, "SageMaker AI Training Job Flow — Fine-Tuning");

// Left: Developer workstation
box(s4, 0.3, 1.2, 2.2, 1.4, C.LIGHT_GRAY, C.GRAY, "", {});
s4.addText("Developer\nWorkstation", {
  x: 0.3, y: 1.2, w: 2.2, h: 0.55, fontSize: 11, bold: true, color: C.DARK,
  fontFace: FONT, align: "center",
});
s4.addText("submit_training.py\nSageMaker Python SDK\nboto3", {
  x: 0.3, y: 1.75, w: 2.2, h: 0.7, fontSize: 8, color: C.GRAY,
  fontFace: FONT, align: "center", lineSpacingMultiple: 1.2,
});

// Arrow: Dev → SageMaker
arrow(s4, 2.5, 1.9, 3.5, 1.9, C.ORANGE, { width: 2 });
label(s4, 2.5, 1.55, 1.2, "CreateTrainingJob\nAPI call", { fontSize: 7, color: C.ORANGE });

// Center: SageMaker Training Job
box(s4, 3.5, 1.0, 5.5, 2.8, C.LIGHT_ORANGE, C.ORANGE, "", {});
s4.addText("Amazon SageMaker AI — Training Job", {
  x: 3.5, y: 1.05, w: 5.5, h: 0.4, fontSize: 12, bold: true, color: C.ORANGE,
  fontFace: FONT, align: "center",
});

// Inside SageMaker: DLC container
box(s4, 3.8, 1.55, 4.9, 0.55, C.WHITE, C.ORANGE,
  "HuggingFace DLC: PyTorch 2.8.0 + Transformers 4.56.2 + CUDA", { fontSize: 8 });

// Inside: train.py flow
const trainSteps = [
  "Load base model\n(QLoRA 4-bit via\nbitsandbytes)",
  "Apply LoRA\nadapter\n(peft)",
  "SFTTrainer\n(trl)\n325 steps",
  "Save adapter\nto /opt/ml/model",
];
trainSteps.forEach((ts, i) => {
  const tx = 3.9 + i * 1.25;
  box(s4, tx, 2.25, 1.1, 0.9, i === 3 ? C.LIGHT_GREEN : C.WHITE, C.ORANGE, ts, { fontSize: 7 });
  if (i < trainSteps.length - 1) {
    arrow(s4, tx + 1.1, 2.7, tx + 1.25, 2.7, C.ORANGE, { width: 1 });
  }
});

// GPU label
box(s4, 3.8, 3.3, 4.9, 0.35, C.LIGHT_GRAY, C.GRAY,
  "GPU: ml.g5.2xlarge (1× A10G 24GB) or ml.g5.12xlarge (4× A10G 96GB)", { fontSize: 8 });

// Right: S3 output
box(s4, 9.8, 1.2, 2.8, 1.0, C.LIGHT_GREEN, C.GREEN,
  "S3: output/\n<model-slug>/\noutput.tar.gz", { fontSize: 9, color: C.GREEN });
arrow(s4, 9.0, 1.9, 9.8, 1.7, C.GREEN, { width: 2 });
label(s4, 9.0, 1.45, 1.0, "Upload\nadapter", { fontSize: 7, color: C.GREEN });

// Top-right: HuggingFace Hub
box(s4, 9.8, 2.8, 2.8, 0.9, C.LIGHT_PURPLE, C.PURPLE,
  "Hugging Face Hub\nBase model weights\n(downloaded at job start)", { fontSize: 8, color: C.PURPLE });
arrow(s4, 9.8, 3.25, 9.0, 2.5, C.PURPLE, { width: 1.5, dash: "dash" });
label(s4, 9.5, 2.45, 1.5, "Download\nweights", { fontSize: 7, color: C.PURPLE });

// Bottom: S3 input
box(s4, 0.3, 3.3, 2.2, 0.7, C.LIGHT_GREEN, C.GREEN,
  "S3: data/\ntrain.jsonl", { fontSize: 9, color: C.GREEN });
arrow(s4, 2.5, 3.65, 3.5, 2.7, C.GREEN, { width: 1.5 });
label(s4, 2.3, 3.2, 1.5, "Input\nchannel", { fontSize: 7, color: C.GREEN });

// Bottom flow summary
const sumY = 4.3;
s4.addShape(pptx.ShapeType.roundRect, {
  x: 0.3, y: sumY, w: 12.7, h: 0.5,
  fill: { color: C.LIGHT_BLUE }, line: { color: C.BLUE, width: 0.5 }, rectRadius: 0.04,
});
s4.addText(
  "Flow:  submit_training.py → SageMaker CreateTrainingJob → DLC spins up → downloads HF model → reads S3 data → trains LoRA → saves adapter to S3",
  { x: 0.3, y: sumY, w: 12.7, h: 0.5, fontSize: 9, color: C.DARK, fontFace: FONT, align: "center", valign: "middle" }
);

// Per-model config table
const tblY = 5.1;
s4.addText("Per-Model Training Configuration", {
  x: 0.3, y: tblY, w: 5, h: 0.35, fontSize: 11, bold: true, color: C.DARK, fontFace: FONT,
});
const hdr = { bold: true, fill: C.ORANGE, color: C.WHITE, fontSize: 9, fontFace: FONT };
const cel = (t, i) => ({ text: t, options: { fill: i % 2 === 0 ? C.WHITE : C.LIGHT_GRAY, fontSize: 9, fontFace: FONT } });
s4.addTable([
  [{ text: "Model", options: hdr }, { text: "Instance", options: hdr },
   { text: "Quantization", options: hdr }, { text: "Steps", options: hdr },
   { text: "Time", options: hdr }, { text: "Cost", options: hdr }],
  [cel("Mistral-Nemo-Base-2407", 0), cel("ml.g5.2xlarge (1× A10G)", 0),
   cel("QLoRA 4-bit", 0), cel("325", 0), cel("~41 min", 0), cel("~$1.31", 0)],
  [cel("Qwen3-14B", 1), cel("ml.g5.12xlarge (4× A10G)", 1),
   cel("QLoRA 4-bit", 1), cel("325", 1), cel("~93 min", 1), cel("~$11.78", 1)],
  [cel("Gemma 3 12B", 0), cel("ml.g5.2xlarge (1× A10G)", 0),
   cel("QLoRA 4-bit", 0), cel("325", 0), cel("~87 min", 0), cel("~$2.49", 0)],
], {
  x: 0.3, y: tblY + 0.4, w: 12.7,
  colW: [2.8, 2.8, 1.5, 0.8, 1.2, 1.0],
  border: { type: "solid", pt: 0.5, color: "D0D7E3" },
});

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — SageMaker Inference Job Flow
// ══════════════════════════════════════════════════════════════════════════════
const s5 = pptx.addSlide();
sectionTitle(s5, "SageMaker AI Inference Job Flow — SLM Batch Prediction");

// Left: Developer
box(s5, 0.3, 1.2, 2.0, 1.2, C.LIGHT_GRAY, C.GRAY, "", {});
s5.addText("Developer", { x: 0.3, y: 1.2, w: 2.0, h: 0.4, fontSize: 11, bold: true, color: C.DARK, fontFace: FONT, align: "center" });
s5.addText("submit_inference.py\nSageMaker SDK", { x: 0.3, y: 1.6, w: 2.0, h: 0.6, fontSize: 8, color: C.GRAY, fontFace: FONT, align: "center" });

arrow(s5, 2.3, 1.8, 3.2, 1.8, C.ORANGE, { width: 2 });

// Center: SageMaker Inference Job
box(s5, 3.2, 1.0, 6.0, 2.6, C.LIGHT_ORANGE, C.ORANGE, "", {});
s5.addText("Amazon SageMaker AI — Inference Job (Training Job API)", {
  x: 3.2, y: 1.05, w: 6.0, h: 0.35, fontSize: 11, bold: true, color: C.ORANGE, fontFace: FONT, align: "center",
});

// Inside: inference_slm.py flow
const infSteps = [
  "Load base model\n(QLoRA 4-bit)",
  "Merge LoRA\nadapter from S3",
  "Run 992 test\nexamples",
  "Save preds\nto output/",
];
infSteps.forEach((is, i) => {
  const ix = 3.4 + i * 1.45;
  box(s5, ix, 1.55, 1.3, 0.85, C.WHITE, C.ORANGE, is, { fontSize: 8 });
  if (i < infSteps.length - 1) {
    arrow(s5, ix + 1.3, 1.97, ix + 1.45, 1.97, C.ORANGE, { width: 1 });
  }
});

// Qwen3 special handling
box(s5, 3.4, 2.6, 5.6, 0.45, C.LIGHT_PURPLE, C.PURPLE,
  "Qwen3: native chat template (<|im_start|>/<|im_end|>) + /no_think + StopOnImEnd", { fontSize: 8, color: C.PURPLE });

// S3 inputs (left-bottom)
box(s5, 0.3, 3.0, 2.0, 0.6, C.LIGHT_GREEN, C.GREEN, "S3: data/\ntest.jsonl", { fontSize: 9, color: C.GREEN });
arrow(s5, 2.3, 3.3, 3.2, 2.2, C.GREEN, { width: 1.5 });
label(s5, 2.0, 2.6, 1.5, "test\nchannel", { fontSize: 7, color: C.GREEN });

box(s5, 0.3, 3.9, 2.0, 0.6, C.LIGHT_GREEN, C.GREEN, "S3: output/\nadapter/", { fontSize: 9, color: C.GREEN });
arrow(s5, 2.3, 4.2, 3.2, 2.8, C.GREEN, { width: 1.5 });
label(s5, 2.0, 3.5, 1.5, "adapter\nchannel", { fontSize: 7, color: C.GREEN });

// S3 output (right)
box(s5, 9.8, 1.3, 2.8, 0.9, C.LIGHT_GREEN, C.GREEN,
  "S3: inference-output/\n<model-slug>/\npreds_*_slm.jsonl", { fontSize: 9, color: C.GREEN });
arrow(s5, 9.2, 1.8, 9.8, 1.75, C.GREEN, { width: 2 });

// HF Hub
box(s5, 9.8, 2.6, 2.8, 0.7, C.LIGHT_PURPLE, C.PURPLE,
  "Hugging Face Hub\nBase model weights", { fontSize: 8, color: C.PURPLE });
arrow(s5, 9.8, 2.95, 9.2, 2.2, C.PURPLE, { width: 1, dash: "dash" });

// Bottom: scoring flow
const scoreY = 5.0;
s5.addText("Post-Inference Scoring Pipeline", {
  x: 0.3, y: scoreY - 0.1, w: 5, h: 0.35, fontSize: 11, bold: true, color: C.DARK, fontFace: FONT,
});
const scoreSteps = [
  { label: "Download\noutput.tar.gz", fill: C.LIGHT_GREEN, border: C.GREEN },
  { label: "Extract\npreds_*.jsonl", fill: C.LIGHT_GRAY, border: C.GRAY },
  { label: "filter.py\nNoise removal", fill: C.LIGHT_BLUE, border: C.BLUE },
  { label: "evaluate.py\nF1/Precision/Recall", fill: C.LIGHT_BLUE, border: C.BLUE },
  { label: "results.json\nAppend metrics", fill: C.LIGHT_TEAL, border: C.TEAL },
  { label: "Upload to S3\nresults/", fill: C.LIGHT_GREEN, border: C.GREEN },
];
scoreSteps.forEach((ss, i) => {
  const sx = 0.3 + i * 2.15;
  box(s5, sx, scoreY + 0.35, 1.9, 0.8, ss.fill, ss.border, ss.label, { fontSize: 8 });
  if (i < scoreSteps.length - 1) {
    arrow(s5, sx + 1.9, scoreY + 0.75, sx + 2.15, scoreY + 0.75, C.GRAY, { width: 1 });
  }
});

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — Bedrock Frontier Model Evaluation Flow
// ══════════════════════════════════════════════════════════════════════════════
const s6 = pptx.addSlide();
sectionTitle(s6, "Amazon Bedrock — Frontier Model Evaluation Flow");

// Left: Developer
box(s6, 0.3, 1.3, 2.0, 1.0, C.LIGHT_GRAY, C.GRAY, "Developer\nevaluate_bedrock.py", { fontSize: 9 });

arrow(s6, 2.3, 1.8, 3.3, 1.8, C.PURPLE, { width: 2 });
label(s6, 2.3, 1.45, 1.2, "boto3\nConverse API", { fontSize: 7, color: C.PURPLE });

// Center: Bedrock
box(s6, 3.3, 1.0, 5.5, 2.5, C.LIGHT_PURPLE, C.PURPLE, "", {});
s6.addText("Amazon Bedrock", {
  x: 3.3, y: 1.05, w: 5.5, h: 0.35, fontSize: 13, bold: true, color: C.PURPLE, fontFace: FONT, align: "center",
});

// Two model boxes inside
box(s6, 3.6, 1.55, 2.4, 0.8, C.WHITE, C.PURPLE, "Claude Opus 4.6\nus.anthropic.\nclaude-opus-4-6-v1", { fontSize: 8 });
box(s6, 6.2, 1.55, 2.4, 0.8, C.WHITE, C.TEAL, "Amazon Nova Pro\namazon.\nnova-pro-v1:0", { fontSize: 8 });

// Prompt strategies
const stratY = 2.55;
const strats = [
  { name: "Zero-shot", desc: "No examples", color: C.BLUE },
  { name: "5-shot", desc: "5 labeled examples", color: C.ORANGE },
  { name: "5-shot CoT", desc: "5 examples + CoT", color: C.GREEN },
];
strats.forEach((st, i) => {
  box(s6, 3.6 + i * 1.75, stratY, 1.6, 0.55, C.WHITE, st.color, `${st.name}\n${st.desc}`, { fontSize: 7, color: st.color });
});

// Right: outputs
box(s6, 9.5, 1.3, 3.0, 0.7, C.LIGHT_GREEN, C.GREEN,
  "results/\npreds_<model>_<strategy>.jsonl", { fontSize: 9, color: C.GREEN });
arrow(s6, 8.8, 1.8, 9.5, 1.65, C.GREEN, { width: 2 });

// Bottom: 6 evaluation runs
const evalY = 3.8;
s6.addText("6 Evaluation Runs (3 strategies × 2 models)", {
  x: 0.3, y: evalY, w: 12, h: 0.35, fontSize: 11, bold: true, color: C.DARK, fontFace: FONT,
});
const runs = [
  { model: "Claude", strat: "zero_shot", f1: "93.45%", color: C.PURPLE },
  { model: "Claude", strat: "five_shot", f1: "98.99%", color: C.PURPLE },
  { model: "Claude", strat: "five_shot_cot", f1: "99.39%", color: C.PURPLE },
  { model: "Nova", strat: "zero_shot", f1: "90.83%", color: C.TEAL },
  { model: "Nova", strat: "five_shot", f1: "97.77%", color: C.TEAL },
  { model: "Nova", strat: "five_shot_cot", f1: "97.16%", color: C.TEAL },
];
runs.forEach((run, i) => {
  const rx = 0.3 + i * 2.15;
  box(s6, rx, evalY + 0.45, 1.95, 0.85, C.WHITE, run.color, "", {});
  s6.addText(run.model, {
    x: rx, y: evalY + 0.48, w: 1.95, h: 0.25,
    fontSize: 9, bold: true, color: run.color, fontFace: FONT, align: "center",
  });
  s6.addText(run.strat, {
    x: rx, y: evalY + 0.72, w: 1.95, h: 0.2,
    fontSize: 7, color: C.GRAY, fontFace: FONT, align: "center",
  });
  s6.addText(`F1: ${run.f1}`, {
    x: rx, y: evalY + 0.95, w: 1.95, h: 0.25,
    fontSize: 10, bold: true, color: C.DARK, fontFace: FONT, align: "center",
  });
});

// System prompt box
const sysY = 5.6;
s6.addText("System Prompt (all runs)", {
  x: 0.3, y: sysY, w: 5, h: 0.3, fontSize: 10, bold: true, color: C.DARK, fontFace: FONT,
});
s6.addShape(pptx.ShapeType.roundRect, {
  x: 0.3, y: sysY + 0.3, w: 12.7, h: 0.7,
  fill: { color: C.LIGHT_GRAY }, line: { color: C.GRAY, width: 0.5 }, rectRadius: 0.04,
});
s6.addText(
  '"You are a 5G core network expert. Analyze 3GPP signaling logs and identify the root cause. ' +
  'Respond with ONLY a JSON array containing one label from: [core_network_failure, authentication_failure, normal, ...]"',
  { x: 0.5, y: sysY + 0.3, w: 12.3, h: 0.7, fontSize: 8, color: C.DARK, fontFace: FONT, align: "left", valign: "middle", italic: true }
);

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — QLoRA Fine-Tuning Architecture (model internals)
// ══════════════════════════════════════════════════════════════════════════════
const s7 = pptx.addSlide();
sectionTitle(s7, "QLoRA 4-bit Fine-Tuning Architecture");

// Base model (frozen)
box(s7, 0.5, 1.2, 4.5, 3.5, C.LIGHT_GRAY, C.GRAY, "", {});
s7.addText("Base Model (Frozen, 4-bit Quantized)", {
  x: 0.5, y: 1.25, w: 4.5, h: 0.35, fontSize: 11, bold: true, color: C.DARK, fontFace: FONT, align: "center",
});
s7.addText("~14B parameters\nLoaded via bitsandbytes NF4\n~6-7 GB VRAM", {
  x: 0.5, y: 1.65, w: 4.5, h: 0.7, fontSize: 9, color: C.GRAY, fontFace: FONT, align: "center",
});

// Transformer layers
const layerNames = ["Embedding", "Attention (Q, K, V, O)", "MLP (gate, up, down)", "LM Head"];
layerNames.forEach((ln, i) => {
  const ly = 2.5 + i * 0.55;
  box(s7, 0.8, ly, 3.9, 0.45, C.WHITE, C.GRAY, ln, { fontSize: 9 });
});

// LoRA adapter (trainable)
box(s7, 5.8, 1.2, 3.5, 3.5, C.LIGHT_ORANGE, C.ORANGE, "", {});
s7.addText("LoRA Adapter (Trainable)", {
  x: 5.8, y: 1.25, w: 3.5, h: 0.35, fontSize: 11, bold: true, color: C.ORANGE, fontFace: FONT, align: "center",
});
s7.addText("< 1% of total parameters\nRank r=16, alpha=32\nTarget: q_proj, v_proj", {
  x: 5.8, y: 1.65, w: 3.5, h: 0.7, fontSize: 9, color: C.GRAY, fontFace: FONT, align: "center",
});

// LoRA math
box(s7, 6.1, 2.5, 2.9, 0.6, C.WHITE, C.ORANGE,
  "W' = W_frozen + B × A\nA: d×r   B: r×d   (r=16)", { fontSize: 9 });

box(s7, 6.1, 3.3, 2.9, 0.5, C.WHITE, C.ORANGE,
  "Only A and B are trained\n~0.5% of total weights", { fontSize: 9 });

box(s7, 6.1, 4.0, 2.9, 0.5, C.LIGHT_GREEN, C.GREEN,
  "Saved as adapter/\n~50-100 MB per model", { fontSize: 9, color: C.GREEN });

// Arrow: frozen → LoRA merge
arrow(s7, 5.0, 3.0, 5.8, 3.0, C.ORANGE, { width: 2 });
label(s7, 5.0, 2.65, 1.0, "Merge at\ninference", { fontSize: 7, color: C.ORANGE });

// Right: Training config
box(s7, 10.0, 1.2, 2.8, 3.5, C.LIGHT_BLUE, C.BLUE, "", {});
s7.addText("Training Config", {
  x: 10.0, y: 1.25, w: 2.8, h: 0.35, fontSize: 11, bold: true, color: C.BLUE, fontFace: FONT, align: "center",
});
const configs = [
  "SFTTrainer (trl)",
  "batch_size: 1",
  "grad_accum: 8",
  "eff. batch: 8",
  "max_steps: 325",
  "lr: 2e-4",
  "scheduler: cosine",
  "bf16: true",
  "max_seq_len: 2048",
];
configs.forEach((cfg, i) => {
  s7.addText(cfg, {
    x: 10.2, y: 1.7 + i * 0.3, w: 2.4, h: 0.28,
    fontSize: 8, color: i === 0 ? C.BLUE : C.DARK, fontFace: FONT, align: "left",
    bold: i === 0,
  });
});

// Bottom: Why QLoRA
const whyY = 5.2;
s7.addShape(pptx.ShapeType.roundRect, {
  x: 0.5, y: whyY, w: 12.3, h: 0.8,
  fill: { color: C.LIGHT_BLUE }, line: { color: C.BLUE, width: 0.5 }, rectRadius: 0.06,
});
s7.addText(
  "Why QLoRA 4-bit?  All 12-14B models exceed 24GB A10G VRAM in BF16 with training overhead. " +
  "QLoRA compresses base weights to 4-bit NF4 (~6-7GB), leaving headroom for activations, gradients, and optimizer states. " +
  "Qwen3-14B additionally needs 4× GPUs due to heavier activation memory.",
  { x: 0.7, y: whyY, w: 11.9, h: 0.8, fontSize: 8, color: C.DARK, fontFace: FONT, align: "left", valign: "middle" }
);

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Data Generation Flow
// ══════════════════════════════════════════════════════════════════════════════
const s8 = pptx.addSlide();
sectionTitle(s8, "Synthetic Data Generation Flow");

// Left: generate_data.py
box(s8, 0.3, 1.3, 2.2, 1.0, C.LIGHT_GRAY, C.GRAY, "generate_data.py\nboto3 Bedrock client", { fontSize: 9 });

arrow(s8, 2.5, 1.8, 3.5, 1.8, C.PURPLE, { width: 2 });
label(s8, 2.5, 1.45, 1.2, "InvokeModel\nAPI", { fontSize: 7, color: C.PURPLE });

// Bedrock
box(s8, 3.5, 1.1, 3.5, 1.5, C.LIGHT_PURPLE, C.PURPLE, "", {});
s8.addText("Amazon Bedrock", { x: 3.5, y: 1.15, w: 3.5, h: 0.3, fontSize: 12, bold: true, color: C.PURPLE, fontFace: FONT, align: "center" });
s8.addText("Claude Opus\nGenerates synthetic 3GPP\nNAS/NGAP/RRC logs", {
  x: 3.5, y: 1.5, w: 3.5, h: 0.9, fontSize: 9, color: C.GRAY, fontFace: FONT, align: "center",
});

arrow(s8, 7.0, 1.8, 8.0, 1.8, C.GREEN, { width: 2 });

// Output JSONL
box(s8, 8.0, 1.1, 3.0, 1.5, C.LIGHT_GREEN, C.GREEN, "", {});
s8.addText("JSONL Output", { x: 8.0, y: 1.15, w: 3.0, h: 0.3, fontSize: 11, bold: true, color: C.GREEN, fontFace: FONT, align: "center" });
s8.addText('{"log": "2024-01-15...",\n "root_cause": ["congestion"]}', {
  x: 8.1, y: 1.5, w: 2.8, h: 0.7, fontSize: 8, color: C.DARK, fontFace: FONT, align: "left",
});

arrow(s8, 9.5, 2.6, 9.5, 3.2, C.GREEN, { width: 1.5 });

// S3 upload
box(s8, 8.0, 3.2, 3.0, 0.7, C.LIGHT_GREEN, C.GREEN, "S3: data/\ntrain.jsonl + test.jsonl", { fontSize: 9, color: C.GREEN });

// 8 failure types
const ftY = 3.2;
s8.addText("8 Failure Types", { x: 0.3, y: ftY, w: 3, h: 0.3, fontSize: 11, bold: true, color: C.DARK, fontFace: FONT });
const failureTypes = [
  "core_network_failure", "authentication_failure", "normal", "handover_failure",
  "congestion", "qos_violation", "transport_jitter", "radio_failure",
];
const ftColors = [C.RED, C.ORANGE, C.GREEN, C.BLUE, C.PURPLE, C.TEAL, C.YELLOW, C.GRAY];
failureTypes.forEach((ft, i) => {
  const col = Math.floor(i / 4);
  const row = i % 4;
  const fx = 0.3 + col * 3.5;
  const fy = ftY + 0.4 + row * 0.45;
  box(s8, fx, fy, 3.2, 0.38, C.WHITE, ftColors[i], ft, { fontSize: 8, color: ftColors[i] });
});

// Dataset stats
const dsY = 5.5;
s8.addShape(pptx.ShapeType.roundRect, {
  x: 0.3, y: dsY, w: 12.4, h: 0.6,
  fill: { color: C.LIGHT_BLUE }, line: { color: C.BLUE, width: 0.5 }, rectRadius: 0.04,
});
s8.addText(
  "Dataset:  train.jsonl = 1,300 examples  |  test.jsonl = 992 examples  |  ~124-125 per failure type  |  Balanced distribution",
  { x: 0.3, y: dsY, w: 12.4, h: 0.6, fontSize: 10, color: C.DARK, fontFace: FONT, align: "center", valign: "middle" }
);

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — Sympathetic Noise Filter Architecture
// ══════════════════════════════════════════════════════════════════════════════
const s9 = pptx.addSlide();
sectionTitle(s9, "Sympathetic Noise Filter — Post-Processing Pipeline");

// Input
box(s9, 0.3, 1.3, 2.5, 1.0, C.LIGHT_GRAY, C.GRAY,
  "Raw Model Output\n(free-form text or JSON)", { fontSize: 9 });

arrow(s9, 2.8, 1.8, 3.5, 1.8, C.BLUE, { width: 2 });

// Stage 1: Extract
box(s9, 3.5, 1.2, 2.8, 1.2, C.LIGHT_BLUE, C.BLUE, "", {});
s9.addText("Stage 1: Extract", { x: 3.5, y: 1.22, w: 2.8, h: 0.3, fontSize: 10, bold: true, color: C.BLUE, fontFace: FONT, align: "center" });
s9.addText("extract_root_cause_from_text()\n• JSON array parsing\n• Exact label matching\n• Fuzzy keyword synonyms\n• <think> block stripping", {
  x: 3.6, y: 1.55, w: 2.6, h: 0.8, fontSize: 7, color: C.DARK, fontFace: FONT, align: "left", lineSpacingMultiple: 1.15,
});

arrow(s9, 6.3, 1.8, 7.0, 1.8, C.BLUE, { width: 2 });

// Stage 2: Filter
box(s9, 7.0, 1.2, 2.8, 1.2, C.LIGHT_ORANGE, C.ORANGE, "", {});
s9.addText("Stage 2: Filter", { x: 7.0, y: 1.22, w: 2.8, h: 0.3, fontSize: 10, bold: true, color: C.ORANGE, fontFace: FONT, align: "center" });
s9.addText("filter_sympathetic_noise()\n• Remove 15 noise codes\n• Normalize to 8 valid labels\n• Default to 'normal'\n  if all filtered out", {
  x: 7.1, y: 1.55, w: 2.6, h: 0.8, fontSize: 7, color: C.DARK, fontFace: FONT, align: "left", lineSpacingMultiple: 1.15,
});

arrow(s9, 9.8, 1.8, 10.5, 1.8, C.GREEN, { width: 2 });

// Output
box(s9, 10.5, 1.3, 2.3, 1.0, C.LIGHT_GREEN, C.GREEN,
  "Clean Labels\n[\"congestion\"]", { fontSize: 9, color: C.GREEN });

// Noise codes grid
const ncY = 3.0;
s9.addText("15 Sympathetic Noise Codes (removed before scoring)", {
  x: 0.3, y: ncY, w: 8, h: 0.35, fontSize: 11, bold: true, color: C.DARK, fontFace: FONT,
});
const noiseCodes = [
  "HEARTBEAT_TIMEOUT", "KEEPALIVE_FAIL", "KEEPALIVE_TIMEOUT",
  "SECONDARY_ALARM", "CASCADING_FAILURE", "PFCP_HEARTBEAT_TIMEOUT",
  "N2_HEARTBEAT_TIMEOUT", "N11_HEARTBEAT_TIMEOUT", "TIMER_EXPIRY",
  "RETRANSMISSION", "DUPLICATE_NAS", "SPURIOUS_MEASUREMENT",
  "BEAM_FAILURE_RECOVERY", "RLC_RETRANSMISSION", "HARQ_NACK",
];
noiseCodes.forEach((nc, i) => {
  const col = Math.floor(i / 5);
  const row = i % 5;
  const nx = 0.3 + col * 4.3;
  const ny = ncY + 0.45 + row * 0.38;
  box(s9, nx, ny, 4.0, 0.32, C.WHITE, C.RED, nc, { fontSize: 8, color: C.RED });
});

// Keyword synonyms
const kwY = 5.7;
s9.addShape(pptx.ShapeType.roundRect, {
  x: 0.3, y: kwY, w: 12.4, h: 0.55,
  fill: { color: C.LIGHT_PURPLE }, line: { color: C.PURPLE, width: 0.5 }, rectRadius: 0.04,
});
s9.addText(
  'Fuzzy keyword synonyms:  "authentication failure" → authentication_failure  |  "QoS flow failed" → qos_violation  |  "radio link failure" → radio_failure  |  etc.',
  { x: 0.5, y: kwY, w: 12.0, h: 0.55, fontSize: 8, color: C.DARK, fontFace: FONT, align: "left", valign: "middle" }
);

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Deployment Architecture (future state)
// ══════════════════════════════════════════════════════════════════════════════
const s10 = pptx.addSlide();
sectionTitle(s10, "Deployment Architecture — Production Patterns");

// Three deployment options side by side
const depOpts = [
  {
    title: "SageMaker Real-Time\nEndpoint",
    x: 0.3, color: C.ORANGE,
    items: [
      "HuggingFaceModel.deploy()",
      "ml.g5.2xlarge (1× A10G)",
      "Auto-scaling",
      "Pay per instance-hour",
      "Managed infrastructure",
    ],
    icon: "Cloud",
  },
  {
    title: "EC2 Self-Hosted",
    x: 4.6, color: C.BLUE,
    items: [
      "g6e.2xlarge (1× L40S)",
      "Full GPU access",
      "Custom serving stack",
      "FastAPI / vLLM / TGI",
      "Manual scaling",
    ],
    icon: "Server",
  },
  {
    title: "AWS Outposts\nEdge / On-Premise",
    x: 8.9, color: C.GREEN,
    items: [
      "Outpost rack in telco DC",
      "Data never leaves facility",
      "Data residency compliance",
      "Low-latency inference",
      "Same APIs as cloud",
    ],
    icon: "Building",
  },
];

depOpts.forEach((d) => {
  // Card
  box(s10, d.x, 1.2, 3.8, 3.8, C.WHITE, d.color, "", {});
  // Header
  s10.addShape(pptx.ShapeType.roundRect, {
    x: d.x, y: 1.2, w: 3.8, h: 0.7,
    fill: { color: d.color }, rectRadius: 0.08,
  });
  s10.addText(d.title, {
    x: d.x, y: 1.2, w: 3.8, h: 0.7,
    fontSize: 12, bold: true, color: C.WHITE, fontFace: FONT, align: "center", valign: "middle",
  });
  // Items
  d.items.forEach((item, i) => {
    s10.addText("•  " + item, {
      x: d.x + 0.2, y: 2.1 + i * 0.42, w: 3.4, h: 0.35,
      fontSize: 9, color: C.DARK, fontFace: FONT, align: "left",
    });
  });
});

// Bottom: model recommendation
const recY = 5.4;
s10.addShape(pptx.ShapeType.roundRect, {
  x: 0.3, y: recY, w: 12.4, h: 0.8,
  fill: { color: C.LIGHT_ORANGE }, line: { color: C.ORANGE, width: 1 }, rectRadius: 0.06,
});
s10.addText(
  "Recommended:  Deploy Mistral-Nemo-Base-2407 (99.70% F1) as primary model.  " +
  "~6-7 GB VRAM with QLoRA 4-bit — fits on a single A10G/L4 GPU.  " +
  "LoRA adapter is ~50-100 MB — fast to download and swap.",
  { x: 0.5, y: recY, w: 12.0, h: 0.8, fontSize: 10, color: C.DARK, fontFace: FONT, align: "left", valign: "middle" }
);

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 11 — IAM and Security Architecture
// ══════════════════════════════════════════════════════════════════════════════
const s11 = pptx.addSlide();
sectionTitle(s11, "IAM Roles and Security Architecture");

// Developer identity
box(s11, 0.3, 1.3, 2.2, 0.8, C.LIGHT_GRAY, C.GRAY, "Developer\nIAM User / SSO", { fontSize: 9 });

// SageMaker execution role
box(s11, 4.0, 1.1, 5.0, 1.5, C.LIGHT_ORANGE, C.ORANGE, "", {});
s11.addText("SageMaker Execution Role", {
  x: 4.0, y: 1.15, w: 5.0, h: 0.35, fontSize: 12, bold: true, color: C.ORANGE, fontFace: FONT, align: "center",
});
s11.addText("arn:aws:iam::ACCOUNT_ID:role/\nAmazonSageMaker-ExecutionRole", {
  x: 4.0, y: 1.5, w: 5.0, h: 0.45, fontSize: 8, color: C.GRAY, fontFace: FONT, align: "center",
});

// Policies
const policies = [
  { name: "AmazonSageMakerFullAccess", color: C.ORANGE },
  { name: "AmazonS3FullAccess", color: C.GREEN },
  { name: "AmazonBedrockFullAccess", color: C.PURPLE },
];
policies.forEach((p, i) => {
  box(s11, 4.2 + i * 1.6, 2.05, 1.5, 0.4, C.WHITE, p.color, p.name, { fontSize: 6, color: p.color });
});

arrow(s11, 2.5, 1.7, 4.0, 1.7, C.GRAY, { width: 1.5 });
label(s11, 2.5, 1.35, 1.5, "Passes role\nto SageMaker", { fontSize: 7, color: C.GRAY });

// Services accessed
const svcY = 3.0;
const services = [
  { name: "Amazon SageMaker AI\nTraining Jobs", x: 0.3, color: C.ORANGE,
    access: "CreateTrainingJob\nDescribeTrainingJob\nCloudWatch Logs" },
  { name: "Amazon S3\nBucket", x: 3.5, color: C.GREEN,
    access: "GetObject (data/)\nPutObject (output/)\nListBucket" },
  { name: "Amazon Bedrock\nConverse API", x: 6.7, color: C.PURPLE,
    access: "InvokeModel\nbedrock:Converse\nModel access enabled" },
  { name: "Hugging Face Hub\n(External)", x: 9.9, color: C.TEAL,
    access: "HTTPS download\nHF_TOKEN env var\nGated model auth" },
];
services.forEach((svc) => {
  box(s11, svc.x, svcY, 2.8, 0.7, svc.color, svc.color, svc.name, {
    fontSize: 10, bold: true, color: C.WHITE });
  s11.addText(svc.access, {
    x: svc.x, y: svcY + 0.8, w: 2.8, h: 0.7,
    fontSize: 8, color: C.DARK, fontFace: FONT, align: "center", lineSpacingMultiple: 1.15,
  });
  // Arrow from role to service
  arrow(s11, 6.5, 2.6, svc.x + 1.4, svcY, C.GRAY, { width: 1, dash: "dash" });
});

// DLC image
const dlcY = 5.2;
s11.addText("Deep Learning Container (DLC) Image", {
  x: 0.3, y: dlcY, w: 6, h: 0.3, fontSize: 11, bold: true, color: C.DARK, fontFace: FONT,
});
s11.addShape(pptx.ShapeType.roundRect, {
  x: 0.3, y: dlcY + 0.35, w: 12.4, h: 0.5,
  fill: { color: C.LIGHT_GRAY }, line: { color: C.GRAY, width: 0.5 }, rectRadius: 0.04,
});
s11.addText(
  "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04",
  { x: 0.5, y: dlcY + 0.35, w: 12.0, h: 0.5, fontSize: 8, color: C.DARK, fontFace: FONT, align: "left", valign: "middle" }
);

// Version pins
const vpY = 6.2;
s11.addShape(pptx.ShapeType.roundRect, {
  x: 0.3, y: vpY, w: 12.4, h: 0.5,
  fill: { color: C.LIGHT_BLUE }, line: { color: C.BLUE, width: 0.5 }, rectRadius: 0.04,
});
s11.addText(
  "Version pins:  PyTorch 2.3.0+cu121 (avoids CUBLAS regression in 2.10+cu128)  |  Transformers 4.46.1+ (Mistral-Nemo support)  |  Python 3.11+",
  { x: 0.5, y: vpY, w: 12.0, h: 0.5, fontSize: 8, color: C.DARK, fontFace: FONT, align: "left", valign: "middle" }
);

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 12 — Component Map (scripts → services)
// ══════════════════════════════════════════════════════════════════════════════
const s12 = pptx.addSlide();
sectionTitle(s12, "Script-to-Service Component Map");

// Left column: Scripts
const scripts = [
  { name: "generate_data.py", desc: "Synthetic data generation", color: C.PURPLE },
  { name: "submit_training.py", desc: "Submit fine-tuning jobs", color: C.ORANGE },
  { name: "train.py", desc: "SageMaker entry point", color: C.ORANGE },
  { name: "submit_inference.py", desc: "Submit inference jobs", color: C.ORANGE },
  { name: "inference_slm.py", desc: "SageMaker entry point", color: C.ORANGE },
  { name: "evaluate_bedrock.py", desc: "Frontier model eval", color: C.PURPLE },
  { name: "evaluate.py", desc: "Scoring pipeline", color: C.BLUE },
  { name: "filter.py", desc: "Noise removal", color: C.BLUE },
  { name: "report_html.py", desc: "HTML report gen", color: C.TEAL },
  { name: "report_ppt.js", desc: "PPTX report gen", color: C.TEAL },
];

// Right column: Services/Libraries
const targets = [
  { name: "Amazon Bedrock", y: 1.4, color: C.PURPLE },
  { name: "SageMaker AI", y: 2.6, color: C.ORANGE },
  { name: "Amazon S3", y: 3.8, color: C.GREEN },
  { name: "Hugging Face Hub", y: 4.6, color: C.TEAL },
  { name: "scikit-learn", y: 5.4, color: C.BLUE },
];

// Draw scripts
scripts.forEach((sc, i) => {
  const sy = 1.2 + i * 0.52;
  box(s12, 0.3, sy, 2.0, 0.42, C.WHITE, sc.color, sc.name, { fontSize: 8, bold: true, color: sc.color });
  label(s12, 2.35, sy + 0.05, 1.8, sc.desc, { fontSize: 7, align: "left" });
});

// Draw targets
targets.forEach((tg) => {
  box(s12, 10.0, tg.y, 2.8, 0.6, tg.color, tg.color, tg.name, {
    fontSize: 11, bold: true, color: C.WHITE });
});

// Connection lines (script index → target index)
const connections = [
  { from: 0, to: 0 }, // generate_data → Bedrock
  { from: 1, to: 1 }, // submit_training → SageMaker
  { from: 2, to: 1 }, // train.py → SageMaker
  { from: 2, to: 3 }, // train.py → HF Hub
  { from: 3, to: 1 }, // submit_inference → SageMaker
  { from: 4, to: 1 }, // inference_slm → SageMaker
  { from: 4, to: 3 }, // inference_slm → HF Hub
  { from: 5, to: 0 }, // evaluate_bedrock → Bedrock
  { from: 6, to: 4 }, // evaluate → scikit-learn
  { from: 1, to: 2 }, // submit_training → S3
  { from: 3, to: 2 }, // submit_inference → S3
  { from: 6, to: 2 }, // evaluate → S3
];

connections.forEach((conn) => {
  const sy = 1.2 + conn.from * 0.52 + 0.21;
  const ty = targets[conn.to].y + 0.3;
  const fromColor = scripts[conn.from].color;
  arrow(s12, 4.2, sy, 10.0, ty, fromColor, { width: 0.8, dash: "dash" });
});

// SDK layer at bottom
const sdkY2 = 6.7;
s12.addShape(pptx.ShapeType.roundRect, {
  x: 0.3, y: sdkY2, w: 12.4, h: 0.45,
  fill: { color: C.LIGHT_GRAY }, line: { color: C.GRAY, width: 0.5 }, rectRadius: 0.04,
});
s12.addText(
  "SDKs:  boto3 (Bedrock, S3)  ·  SageMaker Python SDK (Training Jobs)  ·  transformers + peft + trl (model training)  ·  pptxgenjs (Node.js reports)",
  { x: 0.3, y: sdkY2, w: 12.4, h: 0.45, fontSize: 8, color: C.DARK, fontFace: FONT, align: "center", valign: "middle" }
);

// ══════════════════════════════════════════════════════════════════════════════
// Write file
// ══════════════════════════════════════════════════════════════════════════════
const outPath = "reports/architecture-diagram.pptx";
fs.mkdirSync(path.dirname(outPath), { recursive: true });
pptx.writeFile({ fileName: outPath }).then(() => {
  console.log(`Architecture diagram saved to ${outPath}`);
  console.log(`Slides: ${pptx.slides.length}`);
});
