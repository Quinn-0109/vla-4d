# 报告生成

`VLA-4D-中期报告-2026-09.docx` 是交付物本身；它由这里的两份源文件生成，
**改报告请改源文件再重新生成**，不要直接编辑 docx 后忘了同步回来。

- `中期报告-内容.txt`：全部文字与表格。每行一条指令：
  `TITLE/SUB/H1/H2/H3/P/PN/L/CAP/T/R/GAP/PB`。`T` 给列宽与对齐，其后连续的
  `R` 行是表格内容，用 `|` 分列；正文里 `**...**` 表示加粗。
- `build_docx.js`：排版代码，只负责样式，不含任何文字。

生成（需要 node 与 npm 包 `docx`）：

    npm install docx
    node docs/report/build_docx.js "docs/report/VLA-4D-中期报告-2026-09.docx"

校对（可选，需要 libreoffice-writer 与 poppler-utils）：

    soffice --headless --convert-to pdf --outdir /tmp docs/report/VLA-4D-中期报告-2026-09.docx
    pdftoppm -jpeg -r 90 /tmp/VLA-4D-中期报告-2026-09.pdf /tmp/page

报告中的每个数字都应能在 `docs/05-实验记录.md` 找到出处；
与其他文档冲突时以 `docs/00-索引.md` 的结论台账为准。
