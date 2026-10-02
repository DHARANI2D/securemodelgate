/**
 * Renders a report markdown file to .docx (by default 12_implementation_report.md
 * to SecureModelGate_Implementation_Report.docx).
 * Unlike build_docx.js this IS a converter: the markdown is the single
 * source, so edit the .md and re-run `node build_report_docx.js`.
 * Supports the subset the report uses: YAML title block, # / ## headings,
 * paragraphs with **bold** and `code`, "-" and "1." lists (one nesting
 * level), pipe tables, > blockquotes and ``` code blocks.
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow,
  TableCell, WidthType, ShadingType, BorderStyle, LevelFormat, TableOfContents,
} = require("docx");

// Usage: node build_report_docx.js [source.md output.docx]
// (defaults: the implementation report; also used for 13_developer_checklist_answers.md)
const SRC = path.resolve(__dirname, process.argv[2] || "12_implementation_report.md");
const OUT = path.resolve(__dirname, process.argv[3] || "SecureModelGate_Implementation_Report.docx");
const FONT = "Calibri", MONO = "Consolas", HEAD = "1F3864", RULE = "8EA9C1";
const PAGE_W = 12240 - 2 * 1200;

function inline(text, base = {}) {
  const runs = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) runs.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) runs.push(new TextRun({ text: t.slice(2, -2), font: FONT, bold: true, ...base }));
    else runs.push(new TextRun({ text: t.slice(1, -1), font: MONO, ...base, size: (base.size || 21) - 2 }));
    last = m.index + t.length;
  }
  if (last < text.length) runs.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return runs;
}

function table(rows) {
  const header = rows[0], body = rows.slice(2);
  const ncol = header.length;
  const weights = header.map((_, c) => Math.max(...rows.filter((_, i) => i !== 1).map(r => (r[c] || "").length), 6));
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map(w => Math.max(900, Math.round(PAGE_W * w / total)));
  const scale = PAGE_W / widths.reduce((a, b) => a + b, 0);
  const cols = widths.map(w => Math.floor(w * scale));
  const cell = (text, c, head) => new TableCell({
    width: { size: cols[c], type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: "D9E2F3" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ children: inline(text, { size: 17, bold: head || undefined }) })],
  });
  return new Table({
    width: { size: cols.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: cols,
    rows: [new TableRow({ tableHeader: true, children: header.map((t, c) => cell(t, c, true)) }),
           ...body.map(r => new TableRow({ children: Array.from({ length: ncol }, (_, c) => cell(r[c] || "", c, false)) }))],
  });
}

const raw = fs.readFileSync(SRC, "utf8");
const fm = raw.match(/^---\n([\s\S]*?)\n---\n/);
const meta = {};
fm[1].split("\n").forEach(l => { const m = l.match(/^(\w+):\s*"(.*)"$/); if (m) meta[m[1]] = m[2]; });
const lines = raw.slice(fm[0].length).split("\n");

const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 160 },
    children: [new TextRun({ text: meta.title, font: FONT, bold: true, size: 40, color: HEAD })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: meta.subtitle, font: FONT, italics: true, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: meta.author, font: FONT, size: 21 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 },
    children: [new TextRun({ text: meta.date, font: FONT, size: 18, color: "595959" })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ pageBreakBefore: true, children: [] }),
];

let i = 0, para = [], numInstance = 0, prevWasNumbered = false;
const flush = () => {
  if (para.length) children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 140 },
    children: inline(para.join(" "), { size: 21 }) }));
  para = [];
};
while (i < lines.length) {
  const l = lines[i];
  let m;
  if (l.startsWith("```")) {
    flush(); prevWasNumbered = false; i++;
    while (i < lines.length && !lines[i].startsWith("```")) {
      children.push(new Paragraph({ shading: { type: ShadingType.CLEAR, fill: "F2F2F2" }, spacing: { after: 0 },
        children: [new TextRun({ text: lines[i] || " ", font: MONO, size: 17 })] }));
      i++;
    }
    children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
    i++; continue;
  }
  if ((m = l.match(/^(#{1,2}) (.*)$/))) {
    flush(); prevWasNumbered = false;
    children.push(new Paragraph({ heading: m[1] === "#" ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
      spacing: { before: m[1] === "#" ? 320 : 220, after: 120 },
      border: m[1] === "#" ? { bottom: { color: RULE, space: 4, style: BorderStyle.SINGLE, size: 6 } } : undefined,
      children: [new TextRun({ text: m[2], font: FONT, color: HEAD, bold: true, size: m[1] === "#" ? 30 : 25 })] }));
    i++; continue;
  }
  if (l.startsWith("|")) {
    flush(); prevWasNumbered = false;
    const rows = [];
    while (i < lines.length && lines[i].startsWith("|")) {
      rows.push(lines[i].trim().replace(/^\||\|$/g, "").split("|").map(s => s.trim()));
      i++;
    }
    children.push(table(rows));
    children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
    continue;
  }
  if (l.startsWith(">")) {
    flush(); prevWasNumbered = false;
    const q = [];
    while (i < lines.length && lines[i].startsWith(">")) { q.push(lines[i].replace(/^>\s?/, "")); i++; }
    children.push(new Paragraph({ shading: { type: ShadingType.CLEAR, fill: "F2F5FA" }, indent: { left: 300, right: 300 },
      border: { left: { color: RULE, size: 18, style: BorderStyle.SINGLE, space: 8 } }, spacing: { before: 80, after: 200 },
      children: inline(q.join(" "), { size: 20, italics: true }) }));
    continue;
  }
  if ((m = l.match(/^(\s*)(-|\d+\.) (.*)$/))) {
    flush();
    const level = m[1].length >= 2 ? 1 : 0;
    const numbered = m[2] !== "-";
    if (numbered && level === 0 && !prevWasNumbered) numInstance++;
    let text = m[3]; i++;
    while (i < lines.length && /^\s+\S/.test(lines[i]) && !/^\s*(-|\d+\.) /.test(lines[i])) { text += " " + lines[i].trim(); i++; }
    children.push(new Paragraph({ spacing: { after: 80 }, alignment: AlignmentType.JUSTIFIED,
      numbering: numbered ? { reference: "numbers", level, instance: numInstance } : { reference: "bullets", level },
      children: inline(text, { size: 21 }) }));
    if (level === 0) prevWasNumbered = numbered;
    continue;
  }
  if (l.trim() === "") { flush(); i++; continue; }
  prevWasNumbered = false;
  para.push(l.trim()); i++;
}
flush();

const lvl = (fmt, text) => [0, 1].map(level => ({ level, format: fmt, text: fmt === LevelFormat.BULLET ? (level ? "–" : "•") : `%${level + 1}.`,
  alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360 + level * 360, hanging: 300 } } } }));
const doc = new Document({
  features: { updateFields: true },
  styles: { default: { document: { run: { font: FONT, size: 21 } } } },
  numbering: { config: [{ reference: "bullets", levels: lvl(LevelFormat.BULLET) },
                        { reference: "numbers", levels: lvl(LevelFormat.DECIMAL) }] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 },
    margin: { top: 1080, bottom: 1080, left: 1200, right: 1200 } } }, children }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log("wrote", path.basename(OUT)); });
