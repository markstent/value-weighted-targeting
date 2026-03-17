"""
Generate a non-technical stakeholder summary of the bonus allocation paper.
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import json
import os

OUTPUT_DIR = os.path.dirname(__file__)


def set_style(doc):
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    pf = style.paragraph_format
    pf.space_after = Pt(6)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.15

    for level in range(1, 4):
        heading_style = doc.styles[f'Heading {level}']
        hf = heading_style.font
        hf.name = 'Calibri'
        hf.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
        if level == 1:
            hf.size = Pt(22)
            heading_style.paragraph_format.space_before = Pt(24)
            heading_style.paragraph_format.space_after = Pt(12)
        elif level == 2:
            hf.size = Pt(16)
            heading_style.paragraph_format.space_before = Pt(18)
            heading_style.paragraph_format.space_after = Pt(8)
        elif level == 3:
            hf.size = Pt(13)
            heading_style.paragraph_format.space_before = Pt(12)
            heading_style.paragraph_format.space_after = Pt(6)


def add_styled_table(doc, headers, rows, caption=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Light Shading Accent 1'

    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(10)

    for r, row_data in enumerate(rows):
        for c, cell_text in enumerate(row_data):
            cell = table.rows[r + 1].cells[c]
            cell.text = str(cell_text)
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(10)

    if caption:
        p = doc.add_paragraph()
        run = p.add_run(caption)
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    return table


def add_key_point(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.size = Pt(11)


def add_callout(doc, title, body):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.5)
    p.paragraph_format.right_indent = Cm(1.5)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run(title + '  ')
    run.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
    run = p.add_run(body)
    run.font.size = Pt(11)


def build_document():
    with open(os.path.join(OUTPUT_DIR, 'outputs', 'simulation_results.json')) as f:
        sim = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'outputs', 'hillstrom_results.json')) as f:
        hill = json.load(f)

    doc = Document()
    set_style(doc)

    # Set narrow margins
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # ==================== TITLE PAGE ====================

    for _ in range(6):
        doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Smarter Bonus Targeting')
    run.font.size = Pt(32)
    run.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
    run.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('How Machine Learning Can Improve Bonus ROI\nby Targeting the Right Players')
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Executive Summary')
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    run.italic = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Mark Stent  |  March 2026')
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    doc.add_page_break()

    # ==================== EXECUTIVE SUMMARY ====================

    doc.add_heading('Executive Summary', level=1)

    doc.add_paragraph(
        'Most bonus targeting systems in gaming waste a significant portion of their budget. '
        'They give bonuses to players who would have stayed active anyway, or to players who '
        'are going to leave regardless. This document summarises a new approach that uses '
        'machine learning to identify the specific players whose behaviour actually changes '
        'because of the bonus, and concentrates spending on them.'
    )

    doc.add_paragraph(
        'In simulation testing, the new approach achieves 38% more incremental player '
        'engagement than random bonus distribution, using the same budget. Validation on '
        'a public marketing dataset with 64,000 real customers confirms the approach works '
        'on real-world data.'
    )

    # Key numbers box
    doc.add_paragraph()
    add_callout(doc, 'Bottom line:',
        'For every 100 players given a bonus, the current approach (targeting the most active '
        'players) retains roughly 6 additional players who would otherwise have left. The new '
        'approach retains roughly 9 additional players from the same 100 bonuses. That is '
        '40-50% more retention per bonus dollar spent.')

    doc.add_page_break()

    # ==================== THE PROBLEM ====================

    doc.add_heading('The Problem with Current Bonus Targeting', level=1)

    doc.add_paragraph(
        'Most gaming operators use some form of targeting to decide who receives bonuses. '
        'The most common approach is to identify players who are "most likely to engage" '
        'and give them a bonus. This sounds sensible, but it has a fundamental flaw.'
    )

    doc.add_heading('Not all players respond to bonuses the same way', level=2)

    doc.add_paragraph(
        'When we look at how different types of players react to bonuses, '
        'four distinct patterns emerge:'
    )

    add_styled_table(doc,
        ['Player Type', 'Without Bonus', 'With Bonus', 'Bonus Impact', '% of Players'],
        [
            ['Sure Things', 'Stay active', 'Stay active', 'None - they stay anyway', '~40%'],
            ['Persuadables', 'Might leave', 'Stay active', 'Positive - bonus works', '~15%'],
            ['Lost Causes', 'Will leave', 'Still leave', 'None - nothing helps', '~35%'],
            ['Sleeping Dogs', 'Stay active', 'Might leave', 'Negative - bonus backfires', '~10%'],
        ],
        'Table 1: The four player types and how they respond to bonuses'
    )

    doc.add_paragraph()
    doc.add_paragraph(
        'The problem: traditional targeting systems pick players most likely to be active '
        'after receiving a bonus. But the most active players are Sure Things - they would '
        'have been active regardless. The bonus is wasted on them.'
    )

    doc.add_paragraph(
        'Worse, some players (Sleeping Dogs) actually become less engaged when given a bonus, '
        'perhaps because the bonus triggers a withdrawal or disrupts their natural playing pattern. '
        'Traditional systems sometimes target these players because they look similar to active players.'
    )

    add_callout(doc, 'The key insight:',
        'The goal should not be "who is most likely to engage?" but rather '
        '"whose engagement would change because of the bonus?" These are very different questions.')

    # ==================== THE SOLUTION ====================

    doc.add_heading('The Proposed Solution', level=1)

    doc.add_paragraph(
        'The new framework answers three questions for every player, then combines the answers '
        'to make the best use of the bonus budget:'
    )

    doc.add_heading('Question 1: Will this bonus change their behaviour?', level=2)
    doc.add_paragraph(
        'Instead of predicting who will engage, we predict how much the bonus changes each '
        "player's likelihood of engaging. We do this by running a controlled experiment "
        '(A/B test): randomly give some players a bonus and withhold it from others. Then we '
        'build two separate prediction models - one for the bonus group, one for the control '
        'group - and compare their predictions for each player. The difference tells us the '
        "bonus's impact on that specific player."
    )

    doc.add_heading('Question 2: How valuable is this player?', level=2)
    doc.add_paragraph(
        'A persuadable player who spends $10 per month is less valuable to retain than a '
        'persuadable player who spends $1,000 per month. The framework multiplies the '
        "bonus impact by each player's estimated lifetime value to prioritise high-value "
        'persuadable players.'
    )

    doc.add_heading('Question 3: Is the timing right?', level=2)
    doc.add_paragraph(
        'Players at very low risk of leaving do not need a bonus right now - they will still '
        'be here next month. Players at very high risk of leaving are probably beyond saving. '
        'The framework focuses bonus spend on players at moderate churn risk, where intervention '
        'is both urgent and likely to succeed.'
    )

    doc.add_paragraph()
    doc.add_paragraph(
        'The final priority score for each player is:'
    )

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run('Priority  =  Bonus Impact  x  Player Value  x  Timing Weight')
    run.bold = True
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)

    doc.add_paragraph(
        'Players are ranked by this score, and bonuses are given to the top-ranked players '
        'until the budget runs out. This ensures every bonus dollar is spent where it '
        'generates the most incremental revenue.'
    )

    doc.add_page_break()

    # ==================== RESULTS ====================

    doc.add_heading('What the Results Show', level=1)

    doc.add_heading('Simulation study: 10,000 synthetic players', level=2)

    doc.add_paragraph(
        'We tested the framework in a controlled simulation where we know the ground truth - '
        'we know exactly which players are Persuadable, Sure Things, etc. This lets us '
        'measure precisely how well each targeting approach works.'
    )

    doc.add_paragraph(
        'Four targeting strategies were tested, each with a budget covering 30% of the player base:'
    )

    sr = sim['main_results']
    add_styled_table(doc,
        ['Targeting Strategy', 'Additional Retentions\nper 100 Bonuses', 'Persuadables\nTargeted'],
        [
            ['Random (no targeting)', f'{sr["Random"]["ate_mean"]}', f'{sr["Random"]["pers_mean"]}%'],
            ['Traditional (most active)', f'{sr["Response Model"]["ate_mean"]}', f'{sr["Response Model"]["pers_mean"]}%'],
            ['Uplift model (new)', f'{sr["Uplift Only"]["ate_mean"]}', f'{sr["Uplift Only"]["pers_mean"]}%'],
            ['Full framework (new)', f'{sr["Full Framework"]["ate_mean"]}', f'{sr["Full Framework"]["pers_mean"]}%'],
        ],
        'Table 2: Comparison of targeting strategies (averaged over 50 test runs)'
    )

    doc.add_paragraph()

    doc.add_paragraph(
        f'The uplift model retains {sr["Uplift Only"]["ate_mean"]} additional players per 100 bonuses, '
        f'compared to {sr["Random"]["ate_mean"]} for random targeting - a 38% improvement using the '
        f'same budget. It also captures nearly twice the proportion of Persuadable players '
        f'({sr["Uplift Only"]["pers_mean"]}% vs {sr["Random"]["pers_mean"]}%).'
    )

    doc.add_paragraph(
        f'The traditional approach (targeting the most active players) actually performs worse '
        f'than random ({sr["Response Model"]["ate_mean"]} vs {sr["Random"]["ate_mean"]}). '
        f'This confirms the core problem: targeting active players wastes budget on Sure Things '
        f'and accidentally includes Sleeping Dogs who are harmed by the bonus.'
    )

    doc.add_paragraph(
        'These results were generated at a budget covering 30% of the player base, averaged '
        'over 50 independent replications to ensure statistical reliability.'
    )

    doc.add_heading('Real-world validation: 64,000 e-commerce customers', level=2)

    doc.add_paragraph(
        'To test the approach on real data, we applied it to a public dataset from an email '
        'marketing experiment with 64,000 customers. While this is e-commerce (not gaming), '
        'the targeting problem is identical: which customers should receive a marketing offer?'
    )

    doc.add_paragraph(
        f'The uplift model successfully identified the most responsive customers. The top 10% '
        f'of customers (as ranked by the model) showed an 11.0 percentage point increase in '
        f'engagement, compared to 6.6 percentage points for the overall population. This means '
        f'the model concentrated the marketing effect: the best-targeted customers responded '
        f'67% more strongly than average.'
    )

    doc.add_page_break()

    # ==================== WHAT THIS MEANS ====================

    doc.add_heading('What This Means for the Business', level=1)

    doc.add_heading('Immediate benefits', level=2)

    p = doc.add_paragraph()
    run = p.add_run('More retention per bonus dollar. ')
    run.bold = True
    p.add_run(
        'By targeting Persuadables instead of Sure Things, the same bonus budget retains '
        '38-91% more players (depending on budget level). No additional spend required.'
    )

    p = doc.add_paragraph()
    run = p.add_run('Reduced waste. ')
    run.bold = True
    p.add_run(
        'Bonuses are no longer given to players who would have stayed anyway (Sure Things) '
        'or players who cannot be saved (Lost Causes). Every bonus goes to a player where '
        'it is likely to make a difference.'
    )

    p = doc.add_paragraph()
    run = p.add_run('Avoids accidental harm. ')
    run.bold = True
    p.add_run(
        'The framework identifies and avoids Sleeping Dogs - players who respond negatively '
        'to bonuses. Traditional systems cannot detect these players.'
    )

    doc.add_heading('What is required to implement', level=2)

    p = doc.add_paragraph()
    run = p.add_run('1. A controlled experiment (A/B test). ')
    run.bold = True
    p.add_run(
        'The foundation is a randomised test where some players receive bonuses and others '
        'do not. This must be properly randomised with sufficient sample size (typically '
        '2-4x larger than a standard A/B test).'
    )

    p = doc.add_paragraph()
    run = p.add_run('2. Player data. ')
    run.bold = True
    p.add_run(
        'Historical features such as login frequency, spending patterns, account age, and '
        'bonus redemption history. These must be measured before the experiment starts.'
    )

    p = doc.add_paragraph()
    run = p.add_run('3. Modelling infrastructure. ')
    run.bold = True
    p.add_run(
        'Standard machine learning tools (gradient boosting models). No specialised hardware '
        'or exotic software required. The models can be trained on a standard laptop.'
    )

    p = doc.add_paragraph()
    run = p.add_run('4. Ongoing measurement. ')
    run.bold = True
    p.add_run(
        'A small holdout group (approximately 10% of players) should permanently not receive '
        'model-targeted bonuses, to continuously measure whether the targeting is working.'
    )

    doc.add_heading('Responsible gaming', level=2)

    doc.add_paragraph(
        'The framework includes built-in safeguards. Players flagged by responsible gaming '
        'systems are excluded from bonus targeting before the optimisation runs. This is a '
        'hard constraint, not a soft preference - flagged players never receive a model-directed '
        'bonus. This is essential because an optimisation system could otherwise discover that '
        'vulnerable players are highly responsive to bonuses and target them, which would be '
        'both harmful and a regulatory risk.'
    )

    doc.add_page_break()

    # ==================== NEXT STEPS ====================

    doc.add_heading('Recommended Next Steps', level=1)

    steps = [
        ('Run a pilot A/B test',
         'Design and execute a randomised experiment with a meaningful subset of the player '
         'base. This provides the training data needed to build the uplift model. Aim for '
         'at least 10,000 players split evenly between bonus and control groups.'),
        ('Build and validate the model',
         'Train the uplift model on the pilot data and validate using the metrics described '
         'in the paper (cumulative gain curves, AUUC). Compare predicted uplift rankings '
         'against observed treatment effects.'),
        ('Deploy with measurement',
         'Roll out model-targeted bonuses to the majority of players while maintaining a '
         '10% random treatment group (for retraining data) and a 10% pure control group '
         '(for baseline measurement).'),
        ('Iterate and extend',
         'Monitor incremental revenue per bonus dollar. Retrain models periodically as '
         'player behaviour evolves. Consider extending to personalised bonus amounts '
         '(how much to give, not just whether to give).'),
    ]

    for i, (title, body) in enumerate(steps, 1):
        p = doc.add_paragraph()
        run = p.add_run(f'Step {i}: {title}')
        run.bold = True
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
        doc.add_paragraph(body)

    # ==================== APPENDIX ====================

    doc.add_page_break()
    doc.add_heading('Appendix: Glossary', level=1)

    terms = [
        ('A/B test', 'A controlled experiment where players are randomly split into groups that receive different treatments, allowing us to measure the causal effect of the treatment.'),
        ('Uplift model', 'A machine learning model that predicts how much a treatment (bonus) changes each individual player\'s behaviour, rather than just predicting their behaviour.'),
        ('T-Learner', 'The specific type of uplift model used: two separate models are trained (one on the bonus group, one on the control group) and their predictions are compared.'),
        ('ATE (Average Treatment Effect)', 'The average difference in outcomes between the treated group and the control group, measured in percentage points.'),
        ('Lifetime Value (LTV)', 'The estimated total revenue a player will generate over a defined period (e.g., 90 days), used to prioritise high-value players.'),
        ('AUUC', 'Area Under the Uplift Curve - a single number summarising how well the model identifies high-impact players. Higher is better.'),
        ('Sure Things', 'Players who will engage regardless of whether they receive a bonus. Targeting them wastes budget.'),
        ('Persuadables', 'Players whose behaviour changes because of the bonus. These are the ideal targets.'),
        ('Lost Causes', 'Players who will disengage regardless of bonuses. Targeting them wastes budget.'),
        ('Sleeping Dogs', 'Players who react negatively to bonuses. Targeting them actively harms retention.'),
    ]

    for term, definition in terms:
        p = doc.add_paragraph()
        run = p.add_run(f'{term}: ')
        run.bold = True
        run.font.size = Pt(10)
        run2 = p.add_run(definition)
        run2.font.size = Pt(10)

    # ==================== SAVE ====================

    output_path = os.path.join(OUTPUT_DIR, 'outputs', 'bonus_allocation_stakeholder_summary.docx')
    doc.save(output_path)
    print(f'Stakeholder document saved to: {output_path}')

    # Page count estimate
    total_text = []
    for para in doc.paragraphs:
        total_text.append(para.text)
    word_count = sum(len(t.split()) for t in total_text)
    print(f'Approximate word count: {word_count}')


if __name__ == '__main__':
    build_document()
