"""Generate Jass ML Project Roadmap PDF."""
from fpdf import FPDF
import os

class RoadmapPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, 'Jass ML Project - Roadmap to Superhuman', align='R')
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, title, r=30, g=30, b=30):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(r, g, b)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 30, 30)
        self.set_line_width(0.8)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def metric_row(self, label, value, color=(40, 40, 40)):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(100, 100, 100)
        self.cell(60, 6, label)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(*color)
        self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            w = (self.w - self.l_margin - self.r_margin) / len(headers)
            col_widths = [w] * len(headers)
        # Header
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(45, 45, 55)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align='C')
        self.ln()
        # Rows
        self.set_font('Helvetica', '', 9)
        for ri, row in enumerate(rows):
            if ri % 2 == 0:
                self.set_fill_color(245, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            self.set_text_color(40, 40, 40)
            for i, cell in enumerate(row):
                align = 'L' if i == 0 else 'C'
                self.cell(col_widths[i], 6.5, str(cell), border=1, fill=True, align=align)
            self.ln()
        self.ln(3)

    def phase_block(self, phase, title, status, color, description, target, timeline):
        # Status badge
        r, g, b = color
        self.set_fill_color(r, g, b)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(255, 255, 255)
        badge_w = self.get_string_width(status) + 8
        self.cell(badge_w, 7, status, fill=True)
        self.set_text_color(40, 40, 40)
        self.cell(5, 7, '')
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 7, f'{phase}: {title}', new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.body_text(description)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(200, 30, 30)
        self.cell(20, 6, 'Target: ')
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        self.cell(80, 6, target)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, timeline, align='R', new_x="LMARGIN", new_y="NEXT")
        self.ln(4)


def build_pdf():
    pdf = RoadmapPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── PAGE 1: COVER ──
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font('Helvetica', 'B', 36)
    pdf.set_text_color(30, 30, 40)
    pdf.cell(0, 15, 'Jass ML Project', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font('Helvetica', '', 18)
    pdf.set_text_color(180, 30, 30)
    pdf.cell(0, 10, 'Roadmap to Superhuman Differenzler', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, 'April 2026', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(30)

    # Key stats box
    pdf.set_fill_color(245, 245, 250)
    pdf.set_draw_color(180, 30, 30)
    pdf.set_line_width(0.5)
    box_x = 35
    box_w = pdf.w - 70
    pdf.rect(box_x, pdf.get_y(), box_w, 50, style='DF')
    pdf.set_x(box_x + 10)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(30, 30, 40)
    pdf.cell(box_w - 20, 10, 'Current Status', new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y()
    pdf.set_x(box_x + 10)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(60, 60, 60)
    items = [
        'Arena best:  5.8 avg deviation  (PIMC 500w + C engine)',
        'Live (fixed):  ~10 avg deviation  vs Swisslos training bots',
        'Card play:  8/10  -  Declaration:  5/10',
        'Target:  < 3.5 avg deviation  (superhuman)',
    ]
    for item in items:
        pdf.set_x(box_x + 15)
        pdf.cell(box_w - 30, 8, item, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # ── PAGE 2: ARCHITECTURE ──
    pdf.add_page()
    pdf.section_title('Architecture')
    pdf.body_text(
        'The bot is a Swiss Differenzler Jass AI targeting superhuman play on the Swisslos online platform. '
        'Differenzler is a 4-player trick-taking game where each player predicts their score - '
        'the goal is to minimize |predicted - actual|, not maximize points. Total points per round = 157.'
    )
    pdf.ln(2)
    pdf.sub_title('Tech Stack')
    pdf.table(
        ['Layer', 'Technology', 'Status'],
        [
            ['Game Engine', 'C (gcc -O3, bitmask hands, 196K worlds/sec)', 'Done'],
            ['Search', 'PIMC - 500 worlds, C-accelerated', 'Done'],
            ['Endgame', 'Max^n solver, depth 4 (exact last 4 tricks)', 'Done'],
            ['Beliefs', 'Bayesian tracker over opponent declarations', 'Done'],
            ['Declaration', 'Monte Carlo simulation + percentile pick', 'Needs work'],
            ['Neural Net', 'PyTorch 187K params (66% policy accuracy)', 'POC done'],
            ['Live Play', 'Tampermonkey + Flask + WebSocket bridge', 'Working'],
            ['Data', 'Auto-collection pipeline from Swisslos', 'Working'],
        ],
        col_widths=[35, 85, 50],
    )

    pdf.sub_title('Live Architecture')
    pdf.set_font('Courier', '', 9)
    pdf.set_text_color(60, 60, 60)
    arch = (
        '  Swisslos Browser (WebSocket)\n'
        '       ||\n'
        '  Tampermonkey Extension v7.2.1\n'
        '       || HTTP (localhost:5000)\n'
        '  Flask Server (server.py v1.1)\n'
        '       ||\n'
        '  PIMC Engine + C Engine + Belief Tracker + Endgame Solver'
    )
    pdf.multi_cell(0, 5, arch)
    pdf.ln(5)

    # ── PAGE 3: PERFORMANCE EVOLUTION ──
    pdf.add_page()
    pdf.section_title('Performance Evolution')
    pdf.body_text(
        'Average deviation measures how far off the bot\'s prediction is from actual score each round. '
        'Lower is better. A perfect round has deviation = 0. Strong humans play around 8-10.'
    )
    pdf.table(
        ['Phase', 'What Changed', 'Avg Dev', 'Speed', 'Date'],
        [
            ['1', 'Clean engine + correct rules + PIMC 25w', '9.1', '~5s/round', 'Mar 2026'],
            ['2', 'Median declaration + target-aware opponents', '7.4', '~4s/round', 'Mar 2026'],
            ['3', 'Belief tracking + endgame solver depth 3', '6.8', '~3.5s/round', 'Mar 2026'],
            ['4', 'C engine 500w + calibration + endgame d4', '5.8', '36ms/round', 'Apr 2026'],
            ['4b', 'Neural POC (Python rollouts, too slow)', '6.0', '3s/round', 'Apr 2026'],
            ['5', 'Swisslos deploy (BROKEN seat mapping)', '21.3', '~12ms/move', 'Apr 6'],
            ['5b', 'Seat fix + 500w + 2000 decl worlds', '~10', '~20ms/move', 'Apr 7'],
        ],
        col_widths=[15, 70, 22, 30, 33],
    )

    pdf.sub_title('The Seat Mapping Bug (Found Apr 7)')
    pdf.body_text(
        'The server assumed it was always Swisslos seat 0, but the bot played at seat 2. '
        'This caused: wrong point tracking (tracking Computer 1 instead of us), '
        'belief updates about ourselves instead of opponents, void detection on wrong players, '
        'and wrong opponent points passed to the PIMC search. '
        'Fixing this single bug dropped avg deviation from 21.3 to ~10.'
    )

    pdf.sub_title('Card Play vs Declaration Quality')
    pdf.body_text(
        'Post-fix analysis of 8 rounds shows card play is strong (8/10): the bot correctly tracks '
        'its target, stops taking tricks when over target, and manages point accumulation well. '
        'ALL remaining deviation comes from declaration errors - the simulation opponents '
        'don\'t match how Swisslos bots actually play, causing score predictions to be off.'
    )

    # ── PAGE 4: WHAT WAS TRIED AND FAILED ──
    pdf.add_page()
    pdf.section_title('What Was Tried and Failed')
    pdf.body_text('These experiments are documented to avoid repeating them:')

    failed = [
        ('Smart C Rollout Policy', '5.8 -> 9.1',
         'Added ~150 lines of trick-winning awareness to pick_card. '
         'Distorted score distributions used for declarations. Reverted.'),
        ('ISMCTS (tree search)', '6.3-8.0 vs 5.8',
         '5000 iterations spread across 9 moves x 4 players = too thin. '
         'PIMC concentrates 500 evaluations per card - more robust.'),
        ('More PIMC worlds (2000+)', '5.9 vs 5.8',
         'Diminishing returns. The heuristic rollout quality is the ceiling, not sample count.'),
        ('Neural Python Rollouts', '6.0 at 3s vs 5.8 at 36ms',
         '66% accuracy but 80x slower than C. The C engine\'s quantity advantage wins. '
         'Fix: embed neural net in C via ONNX.'),
        ('Percentile Recalibration', '9.9 vs 10.2',
         'Sweeping percentile/offset against 8 live rounds. Marginal improvement - '
         'the problem is the simulation opponent model, not the percentile.'),
    ]
    for name, result, desc in failed:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(90, 6, name)
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(180, 30, 30)
        pdf.cell(0, 6, result, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 5, desc)
        pdf.ln(3)

    # ── PAGE 5: ROADMAP ──
    pdf.add_page()
    pdf.section_title('Roadmap')
    pdf.ln(2)

    pdf.phase_block(
        'Phase A', 'Declaration Correction Model', 'NEXT',
        (230, 160, 30),
        'Collect 200+ rounds against Swisslos bots (running overnight). '
        'Build a regression model: given hand features (trump count, Nell/Puur, raw points, aces, voids), '
        'learn the systematic bias between simulation prediction and actual Swisslos score. '
        'Apply as a correction layer on top of the existing declaration model.',
        '7-8 avg deviation live',
        '1-2 weeks'
    )

    pdf.phase_block(
        'Phase B', 'ONNX Neural Rollouts in C', 'PLANNED',
        (50, 120, 200),
        'Export the trained PyTorch model to ONNX. Embed ONNX Runtime in jass_engine.c, '
        'replacing pick_card with neural inference during rollouts. '
        '500 worlds x neural-quality play instead of heuristic play. '
        'This fixes the ROOT CAUSE: simulation opponents play realistically, '
        'so declarations become accurate and card play improves further.',
        '5-6 avg deviation live',
        '2-4 weeks'
    )

    pdf.phase_block(
        'Phase C', 'Scaled Training on Real Data', 'PLANNED',
        (50, 120, 200),
        'Generate 100K self-play rounds with ONNX-speed neural rollouts. '
        'Add 1000+ real Swisslos games as additional training signal. '
        'Retrain network with soft labels (per-card PIMC utilities). '
        'Real human data teaches patterns that don\'t exist in self-play.',
        '4-5 avg deviation live',
        '2-3 weeks after Phase B'
    )

    pdf.phase_block(
        'Phase D', 'Opponent Modeling + Meta-game', 'FUTURE',
        (120, 120, 120),
        'Cluster opponent play styles from collected data. '
        'Neural inference of opponent declarations from play patterns. '
        'Meta-game: adjust risk based on cumulative standings '
        '(play safe when leading, take risks when behind). '
        'Continuous improvement loop: play, collect, retrain, deploy.',
        '< 3.5 avg deviation - SUPERHUMAN',
        '3-4 months total'
    )

    # ── PAGE 6: TIMELINE ──
    pdf.add_page()
    pdf.section_title('Timeline to Superhuman')
    pdf.ln(2)

    pdf.table(
        ['Milestone', 'Avg Dev', 'Perfect %', 'vs Humans', 'When'],
        [
            ['Arena baseline (Phase 4)', '5.8', '20%', '(self-play only)', 'Done'],
            ['Live deploy + seat fix', '~10', '~15%', 'Competitive', 'Apr 7'],
            ['Declaration correction (A)', '7-8', '~20%', 'Beats most', '1-2 weeks'],
            ['ONNX neural rollouts (B)', '5-6', '~25%', 'Beats strong', '1-2 months'],
            ['Scaled training (C)', '4-5', '~28%', 'Top tier', '2-3 months'],
            ['Superhuman (D)', '< 3.5', '~35%', 'Beats everyone', '3-4 months'],
        ],
        col_widths=[50, 20, 22, 40, 38],
    )

    pdf.ln(5)
    pdf.sub_title('Key Principles (Validated by Experience)')
    principles = [
        ('Speed enables everything.',
         'The C engine\'s 196K worlds/sec is the foundation. Every improvement multiplies on top.'),
        ('Consistency > intelligence.',
         'A smarter rollout that distorts distributions is WORSE than a dumb consistent one.'),
        ('Declaration is half the game.',
         'A perfect card player with a bad declaration still loses.'),
        ('PIMC beats ISMCTS here.',
         '4-player imperfect info makes tree search iterations too thin.'),
        ('Rollout quality is the ceiling.',
         'More worlds don\'t help once the heuristic converges. Neural rollouts are the unlock.'),
        ('Deploy early, iterate live.',
         'Real-game data is more valuable than theoretical improvements.'),
    ]
    for title, desc in principles:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 5, desc)
        pdf.ln(2)

    # ── PAGE 7: WHAT SUPERHUMAN LOOKS LIKE ──
    pdf.add_page()
    pdf.section_title('What Superhuman Looks Like')
    pdf.ln(2)
    pdf.body_text(
        'The Swisslos Differenzler leaderboard has players with thousands of games. '
        'The best humans average around 7-8 deviation over hundreds of rounds. '
        '"Superhuman" means consistently beating these players:'
    )
    pdf.ln(2)
    pdf.table(
        ['Player Type', 'Avg Deviation', 'Perfect Rounds', 'Key Weakness'],
        [
            ['Casual human', '12-15', '~5%', 'Inconsistent declarations'],
            ['Good human', '8-10', '~10%', 'Misreads opponent patterns'],
            ['Strong human', '7-8', '~15%', 'Occasional misjudgments'],
            ['Our bot (now)', '~10', '~15%', 'Declaration model'],
            ['Our bot (Phase B)', '5-6', '~25%', 'Novel situations'],
            ['Our bot (Phase D)', '< 3.5', '~35%', 'None identified'],
        ],
        col_widths=[38, 30, 30, 72],
    )

    pdf.ln(5)
    pdf.sub_title('The Endgame')
    pdf.body_text(
        'At < 3.5 avg deviation with 35% perfect rounds, the bot would find lines humans miss '
        'and exploit tendencies humans can\'t track. It would adapt to each opponent\'s style, '
        'adjust risk based on match standings, and play optimally from any position. '
        'The card play engine is already most of the way there - the remaining journey '
        'is about making the prediction model match reality.'
    )
    pdf.ln(5)
    pdf.set_font('Helvetica', 'I', 11)
    pdf.set_text_color(180, 30, 30)
    pdf.cell(0, 10, 'The path is clear. Speed got us here. Intelligence gets us the rest of the way.',
             align='C', new_x="LMARGIN", new_y="NEXT")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), 'Jass_ML_Roadmap.pdf')
    pdf.output(out_path)
    print(f'Saved: {out_path}')
    print(f'Pages: {pdf.page_no()}')
    return out_path

if __name__ == '__main__':
    build_pdf()
