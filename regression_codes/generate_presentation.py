"""
Mortgage Default Risk Modeling - PowerPoint Generator
This script generates a professional PowerPoint presentation from the mortgage default modeling project.

Requirements:
pip install python-pptx

Usage:
python generate_presentation.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def add_gradient_background(slide, color1, color2):
    """Add a gradient background to a slide"""
    background = slide.background
    fill = background.fill
    fill.gradient()
    fill.gradient_angle = 45.0
    fill.gradient_stops[0].color.rgb = color1
    fill.gradient_stops[1].color.rgb = color2

def create_rounded_rectangle(slide, left, top, width, height, fill_color, border_color=None, border_width=0):
    """Create a rounded rectangle shape"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(border_width)
    else:
        shape.line.fill.background()
    
    return shape

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    # Enhanced color palette
    BLUE_900 = RGBColor(30, 58, 138)
    BLUE_800 = RGBColor(30, 64, 175)
    BLUE_600 = RGBColor(37, 99, 235)
    BLUE_500 = RGBColor(59, 130, 246)
    BLUE_300 = RGBColor(147, 197, 253)
    BLUE_200 = RGBColor(191, 219, 254)
    BLUE_100 = RGBColor(219, 234, 254)
    BLUE_50 = RGBColor(239, 246, 255)
    
    PURPLE_600 = RGBColor(147, 51, 234)
    PURPLE_500 = RGBColor(168, 85, 247)
    PURPLE_100 = RGBColor(243, 232, 255)
    
    PINK_600 = RGBColor(219, 39, 119)
    PINK_500 = RGBColor(236, 72, 153)
    
    GREEN_600 = RGBColor(22, 163, 74)
    GREEN_500 = RGBColor(34, 197, 94)
    GREEN_100 = RGBColor(220, 252, 231)
    
    RED_600 = RGBColor(220, 38, 38)
    RED_500 = RGBColor(239, 68, 68)
    
    ORANGE_600 = RGBColor(234, 88, 12)
    ORANGE_500 = RGBColor(249, 115, 22)
    ORANGE_100 = RGBColor(255, 237, 213)
    
    SLATE_900 = RGBColor(15, 23, 42)
    SLATE_800 = RGBColor(30, 41, 59)
    SLATE_700 = RGBColor(51, 65, 85)
    SLATE_600 = RGBColor(71, 85, 105)
    SLATE_200 = RGBColor(226, 232, 240)
    SLATE_100 = RGBColor(241, 245, 249)
    SLATE_50 = RGBColor(248, 250, 252)
    
    WHITE = RGBColor(255, 255, 255)
    
    # ========== SLIDE 1: TITLE SLIDE ==========
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_gradient_background(slide, BLUE_900, BLUE_800)
    
    # Icon placeholder (using a circle)
    icon_circle = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(4.3), Inches(1.2), Inches(1.4), Inches(1.4)
    )
    icon_circle.fill.solid()
    icon_circle.fill.fore_color.rgb = RGBColor(255, 255, 255)
    icon_circle.fill.transparency = 0.9
    icon_circle.line.fill.background()
    
    icon_text = icon_circle.text_frame
    icon_text.text = "📊"
    icon_para = icon_text.paragraphs[0]
    icon_para.font.size = Pt(60)
    icon_para.alignment = PP_ALIGN.CENTER
    icon_text.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(1), Inches(3), Inches(8), Inches(1))
    title_frame = title_box.text_frame
    title_frame.text = "Credit Risk Modeling"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(60)
    title_para.font.bold = True
    title_para.font.color.rgb = WHITE
    title_para.alignment = PP_ALIGN.CENTER
    
    # Subtitle
    subtitle_box = slide.shapes.add_textbox(Inches(1), Inches(4.1), Inches(8), Inches(0.8))
    subtitle_frame = subtitle_box.text_frame
    subtitle_frame.text = "Predicting Mortgage Defaults in MBS Pools"
    subtitle_para = subtitle_frame.paragraphs[0]
    subtitle_para.font.size = Pt(32)
    subtitle_para.font.color.rgb = BLUE_200
    subtitle_para.alignment = PP_ALIGN.CENTER
    
    # Divider line
    divider = slide.shapes.add_connector(1, Inches(3), Inches(5.3), Inches(7), Inches(5.3))
    divider.line.color.rgb = BLUE_600
    divider.line.width = Pt(2)
    divider.line.transparency = 0.5
    
    # Author info
    author_box = slide.shapes.add_textbox(Inches(2), Inches(5.6), Inches(6), Inches(1.2))
    author_frame = author_box.text_frame
    author_frame.text = "Divija Rao Balasankula\nFINANCIAL MATHEMATICS\nNovember 19, 2025"
    for para in author_frame.paragraphs:
        para.font.size = Pt(20)
        para.font.color.rgb = BLUE_300
        para.alignment = PP_ALIGN.CENTER
        para.space_after = Pt(6)
    
    # ========== SLIDE 2: THE CHALLENGE ==========
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_gradient_background(slide, SLATE_50, SLATE_100)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(9), Inches(0.7))
    title_frame = title_box.text_frame
    title_frame.text = "The Challenge"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(48)
    title_para.font.bold = True
    title_para.font.color.rgb = SLATE_800
    
    # Market Scale Box
    market_box = create_rounded_rectangle(
        slide, Inches(0.8), Inches(1.8), Inches(4), Inches(2.2),
        WHITE, RED_500, 4
    )
    market_text = market_box.text_frame
    market_text.text = "🚨 Market Scale\n\n$12 Trillion\n\nMBS Market Size"
    market_text.paragraphs[0].font.size = Pt(22)
    market_text.paragraphs[0].font.bold = True
    market_text.paragraphs[0].font.color.rgb = SLATE_800
    market_text.paragraphs[0].alignment = PP_ALIGN.CENTER
    
    market_text.paragraphs[2].font.size = Pt(48)
    market_text.paragraphs[2].font.bold = True
    market_text.paragraphs[2].font.color.rgb = RED_600
    market_text.paragraphs[2].alignment = PP_ALIGN.CENTER
    
    market_text.paragraphs[4].font.size = Pt(18)
    market_text.paragraphs[4].font.color.rgb = SLATE_600
    market_text.paragraphs[4].alignment = PP_ALIGN.CENTER
    market_text.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    # Crisis Impact Box
    crisis_box = create_rounded_rectangle(
        slide, Inches(5.2), Inches(1.8), Inches(4), Inches(2.2),
        WHITE, ORANGE_500, 4
    )
    crisis_text = crisis_box.text_frame
    crisis_text.text = "⚠️ Crisis Impact\n\n2008 Financial Crisis\n\nTriggered by underestimated defaults"
    crisis_text.paragraphs[0].font.size = Pt(22)
    crisis_text.paragraphs[0].font.bold = True
    crisis_text.paragraphs[0].font.color.rgb = SLATE_800
    crisis_text.paragraphs[0].alignment = PP_ALIGN.CENTER
    
    crisis_text.paragraphs[2].font.size = Pt(26)
    crisis_text.paragraphs[2].font.bold = True
    crisis_text.paragraphs[2].font.color.rgb = ORANGE_600
    crisis_text.paragraphs[2].alignment = PP_ALIGN.CENTER
    
    crisis_text.paragraphs[4].font.size = Pt(18)
    crisis_text.paragraphs[4].font.color.rgb = SLATE_600
    crisis_text.paragraphs[4].alignment = PP_ALIGN.CENTER
    crisis_text.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    # Goal Box - Gradient effect with shadow
    goal_box = create_rounded_rectangle(
        slide, Inches(1.5), Inches(4.8), Inches(7), Inches(1.2),
        BLUE_600
    )
    goal_text = goal_box.text_frame
    goal_text.text = "🎯 Goal: Predict loan-level defaults with accuracy and transparency"
    goal_para = goal_text.paragraphs[0]
    goal_para.font.size = Pt(26)
    goal_para.font.bold = True
    goal_para.font.color.rgb = WHITE
    goal_para.alignment = PP_ALIGN.CENTER
    goal_text.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    # Add shadow effect
    goal_box.shadow.inherit = False
    
    # ========== SLIDE 3: APPROACH ==========
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_gradient_background(slide, SLATE_50, SLATE_100)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(9), Inches(0.7))
    title_frame = title_box.text_frame
    title_frame.text = "Approach: Logistic Regression"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(48)
    title_para.font.bold = True
    title_para.font.color.rgb = SLATE_800
    
    reasons = [
        ("Regulatory Acceptance", "Interpretable for compliance"),
        ("Coefficient Explainability", "Clear risk drivers"),
        ("Production Ready", "Easy calibration & deployment")
    ]
    
    y_pos = 1.9
    for idx, (reason, detail) in enumerate(reasons):
        # Reason box with left border accent
        box = create_rounded_rectangle(
            slide, Inches(1.5), Inches(y_pos), Inches(7), Inches(1.1),
            WHITE, GREEN_500, 0
        )
        
        # Add left accent bar
        accent = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(1.5), Inches(y_pos), Inches(0.08), Inches(1.1)
        )
        accent.fill.solid()
        accent.fill.fore_color.rgb = GREEN_500
        accent.line.fill.background()
        
        # Checkmark circle
        check_circle = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            Inches(1.9), Inches(y_pos + 0.3), Inches(0.5), Inches(0.5)
        )
        check_circle.fill.solid()
        check_circle.fill.fore_color.rgb = GREEN_100
        check_circle.line.fill.background()
        
        check_text = check_circle.text_frame
        check_text.text = "✓"
        check_para = check_text.paragraphs[0]
        check_para.font.size = Pt(24)
        check_para.font.color.rgb = GREEN_600
        check_para.alignment = PP_ALIGN.CENTER
        check_text.vertical_anchor = MSO_ANCHOR.MIDDLE
        
        # Text content
        text_box = slide.shapes.add_textbox(Inches(2.6), Inches(y_pos + 0.15), Inches(5.6), Inches(0.8))
        text_frame = text_box.text_frame
        text_frame.text = f"{reason}\n{detail}"
        text_frame.paragraphs[0].font.size = Pt(24)
        text_frame.paragraphs[0].font.bold = True
        text_frame.paragraphs[0].font.color.rgb = SLATE_800
        text_frame.paragraphs[1].font.size = Pt(18)
        text_frame.paragraphs[1].font.color.rgb = SLATE_600
        
        y_pos += 1.5
    
    # ========== SLIDE 4: DATA OVERVIEW ==========
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_gradient_background(slide, SLATE_50, SLATE_100)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(9), Inches(0.7))
    title_frame = title_box.text_frame
    title_frame.text = "Data: Freddie Mac Loan Performance"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(44)
    title_para.font.bold = True
    title_para.font.color.rgb = SLATE_800
    
    data_points = [
        ("20.32M", "Records", "💾", BLUE_500, BLUE_600),
        ("20", "Features\nEngineered", "📊", PURPLE_500, PURPLE_600),
        ("0.8-1%", "Default\nRate", "🎯", PINK_500, PINK_600)
    ]
    
    x_pos = 1.2
    for value, label, emoji, color1, color2 in data_points:
        # Gradient box
        box = create_rounded_rectangle(
            slide, Inches(x_pos), Inches(2), Inches(2.3), Inches(2.2),
            color1
        )
        box.shadow.inherit = False
        
        # Emoji
        emoji_text = slide.shapes.add_textbox(Inches(x_pos + 0.8), Inches(2.2), Inches(0.7), Inches(0.5))
        emoji_frame = emoji_text.text_frame
        emoji_frame.text = emoji
        emoji_para = emoji_frame.paragraphs[0]
        emoji_para.font.size = Pt(40)
        emoji_para.alignment = PP_ALIGN.CENTER
        
        # Value and label
        text_box = slide.shapes.add_textbox(Inches(x_pos + 0.2), Inches(2.8), Inches(1.9), Inches(1.2))
        text_frame = text_box.text_frame
        text_frame.text = f"{value}\n\n{label}"
        text_frame.paragraphs[0].font.size = Pt(40)
        text_frame.paragraphs[0].font.bold = True
        text_frame.paragraphs[0].font.color.rgb = WHITE
        text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        
        text_frame.paragraphs[2].font.size = Pt(16)
        text_frame.paragraphs[2].font.color.rgb = WHITE
        text_frame.paragraphs[2].alignment = PP_ALIGN.CENTER
        text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        
        x_pos += 2.6
    
    # Info banner
    info_box = create_rounded_rectangle(
        slide, Inches(1), Inches(5), Inches(8), Inches(0.9),
        SLATE_800
    )
    info_text = info_box.text_frame
    info_text.text = "📅 Period: 2013-2024  |  ⚠️ Challenge: Extreme Class Imbalance (1:99)"
    info_para = info_text.paragraphs[0]
    info_para.font.size = Pt(22)
    info_para.font.bold = True
    info_para.font.color.rgb = WHITE
    info_para.alignment = PP_ALIGN.CENTER
    info_text.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    # ========== SLIDE 5: VALIDATION STRATEGY ==========
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_gradient_background(slide, SLATE_50, SLATE_100)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(9), Inches(0.7))
    title_frame = title_box.text_frame
    title_frame.text = "Rigorous Validation Strategy"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(48)
    title_para.font.bold = True
    title_para.font.color.rgb = SLATE_800
    
    splits = [
        ("Training", "≤ Dec 2023", "7.87M records", BLUE_500),
        ("Validation", "Jan-Jun 2024", "295K records", PURPLE_500),
        ("Test", "Jul 2024+", "12.2M records", PINK_500)
    ]
    
    y_pos = 1.9
    for split_name, period, size, color in splits:
        # Label
        label_box = slide.shapes.add_textbox(Inches(0.8), Inches(y_pos + 0.15), Inches(1.8), Inches(0.6))
        label_text = label_box.text_frame
        label_text.text = split_name
        label_para = label_text.paragraphs[0]
        label_para.font.size = Pt(24)
        label_para.font.bold = True
        label_para.font.color.rgb = SLATE_800
        label_para.alignment = PP_ALIGN.RIGHT
        
        # Box with colored left border
        box = create_rounded_rectangle(
            slide, Inches(2.8), Inches(y_pos), Inches(6.2), Inches(0.9),
            WHITE
        )
        
        # Colored accent bar
        accent = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(2.8), Inches(y_pos), Inches(0.12), Inches(0.9)
        )
        accent.fill.solid()
        accent.fill.fore_color.rgb = color
        accent.line.fill.background()
        
        # Text content
        text_box = slide.shapes.add_textbox(Inches(3.1), Inches(y_pos + 0.15), Inches(5.6), Inches(0.6))
        text_frame = text_box.text_frame
        text_frame.text = f"{period}          {size}"
        text_para = text_frame.paragraphs[0]
        text_para.font.size = Pt(20)
        text_para.font.bold = True
        text_para.font.color.rgb = SLATE_700
        
        y_pos += 1.3
    
    # Key strength banner
    strength_box = create_rounded_rectangle(
        slide, Inches(1), Inches(5.7), Inches(8), Inches(0.9),
        BLUE_100, BLUE_500, 3
    )
    strength_text = strength_box.text_frame
    strength_text.text = "✓ Key Strength: True out-of-time testing on completely unseen future data"
    strength_para = strength_text.paragraphs[0]
    strength_para.font.size = Pt(20)
    strength_para.font.bold = True
    strength_para.font.color.rgb = BLUE_900
    strength_para.alignment = PP_ALIGN.LEFT
    strength_text.vertical_anchor = MSO_ANCHOR.MIDDLE
    strength_text.margin_left = Inches(0.3)
    
    # ========== SLIDE 6: FEATURE ENGINEERING ==========
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_gradient_background(slide, SLATE_50, SLATE_100)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(9), Inches(0.7))
    title_frame = title_box.text_frame
    title_frame.text = "Feature Engineering"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(48)
    title_para.font.bold = True
    title_para.font.color.rgb = SLATE_800
    
    features = [
        ("Borrower Profile", ["Credit Score", "Original LTV", "DTI Ratio", "Number of Borrowers"]),
        ("Loan Characteristics", ["Interest Rate", "Loan Age", "Remaining Term Ratio"]),
        ("Risk Flags", ["High DTI (>45%)", "High LTV (>95%)", "Low FICO (<660)", "Negative Equity"]),
        ("Macro at Origination", ["30-yr Mortgage Rate", "Unemployment", "HPI YoY Change", "Fed Funds Rate"])
    ]
    
    positions = [(0.5, 1.5), (5.2, 1.5), (0.5, 4.2), (5.2, 4.2)]
    colors = [BLUE_600, PURPLE_600, GREEN_600, ORANGE_600]
    
    for (cat_name, feat_list), (x, y), color in zip(features, positions, colors):
        box = create_rounded_rectangle(
            slide, Inches(x), Inches(y), Inches(4.3), Inches(2.3),
            WHITE, color, 4
        )
        
        # Category title with background
        cat_bg = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x + 0.2), Inches(y + 0.2), Inches(3.9), Inches(0.5)
        )
        cat_bg.fill.solid()
        cat_bg.fill.fore_color.rgb = color
        cat_bg.line.fill.background()
        
        cat_text = cat_bg.text_frame
        cat_text.text = cat_name
        cat_para = cat_text.paragraphs[0]
        cat_para.font.size = Pt(18)
        cat_para.font.bold = True
        cat_para.font.color.rgb = WHITE
        cat_para.alignment = PP_ALIGN.CENTER
        cat_text.vertical_anchor = MSO_ANCHOR.MIDDLE
        
        # Features list
        feat_text = slide.shapes.add_textbox(Inches(x + 0.3), Inches(y + 0.9), Inches(3.7), Inches(1.2))
        feat_frame = feat_text.text_frame
        feat_frame.text = "\n".join([f"• {f}" for f in feat_list])
        for para in feat_frame.paragraphs:
            para.font.size = Pt(14)
            para.font.color.rgb = SLATE_700
            para.space_after = Pt(4)
    
    # ========== SLIDE 7: MODELING PIPELINE ==========
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_gradient_background(slide, SLATE_50, SLATE_100)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(9), Inches(0.7))
    title_frame = title_box.text_frame
    title_frame.text = "Modeling Pipeline"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(48)
    title_para.font.bold = True
    title_para.font.color.rgb = SLATE_800
    
    steps = [
        ("1", "Preprocessing", "Median imputation → Standard scaling → One-hot encoding"),
        ("2", "Model Training", "Logistic Regression with balanced class weights + L1/L2 regularization"),
        ("3", "Hyperparameter Tuning", "Cross-validated grid search on training data"),
        ("4", "Calibration", "Platt scaling on validation set for reliable probabilities")
    ]
    
    y_pos = 1.7
    for num, step_name, detail in steps:
        # Number circle with gradient
        circle = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            Inches(1.2), Inches(y_pos + 0.1), Inches(0.6), Inches(0.6)
        )
        circle.fill.solid()
        circle.fill.fore_color.rgb = BLUE_600
        circle.line.fill.background()
        circle.shadow.inherit = False
        
        num_text = circle.text_frame
        num_text.text = num
        num_para = num_text.paragraphs[0]
        num_para.font.size = Pt(28)
        num_para.font.bold = True
        num_para.font.color.rgb = WHITE
        num_para.alignment = PP_ALIGN.CENTER
        num_text.vertical_anchor = MSO_ANCHOR.MIDDLE
        
        # Step box
        box = create_rounded_rectangle(
            slide, Inches(2), Inches(y_pos), Inches(7), Inches(0.85),
            WHITE
        )
        box.shadow.inherit = False
        
        # Step content
        text_box = slide.shapes.add_textbox(Inches(2.2), Inches(y_pos + 0.1), Inches(6.6), Inches(0.65))
        text_frame = text_box.text_frame
        text_frame.text = f"{step_name}\n{detail}"
        text_frame.paragraphs[0].font.size = Pt(20)
        text_frame.paragraphs[0].font.bold = True
        text_frame.paragraphs[0].font.color.rgb = SLATE_800
        text_frame.paragraphs[1].font.size = Pt(15)
        text_frame.paragraphs[1].font.color.rgb = SLATE_600
        
        # Arrow (except for last step)
        if num != "4":
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.DOWN_ARROW,
                Inches(1.4), Inches(y_pos + 0.9), Inches(0.2), Inches(0.3)
            )
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = BLUE_500
            arrow.line.fill.background()
        
        y_pos += 1.3
    
    # ========== SLIDE 8: RESULTS ==========
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_gradient_background(slide, SLATE_50, SLATE_100)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(9), Inches(0.7))
    title_frame = title_box.text_frame
    title_frame.text = "Out-of-Time Performance (H2 2024+)"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(42)
    title_para.font.bold = True
    title_para.font.color.rgb = SLATE_800
    
    metrics = [
        ("70.05%", "ROC-AUC", GREEN_500, GREEN_600),
        ("4.66%", "Precision\n@ 95th %ile", BLUE_500, BLUE_600),
        ("23.8%", "Recall\n@ 95th %ile", PURPLE_500, PURPLE_600),
        ("6×", "Lift vs\nBaseline", ORANGE_500, ORANGE_600)
    ]
    
    x_pos = 1
    for value, label, color1, color2 in metrics:
        box = create_rounded_rectangle(
            slide, Inches(x_pos), Inches(1.9), Inches(2), Inches(2),
            WHITE, color1, 5
        )
        box.shadow.inherit = False
        
        # Value
        val_text = slide.shapes.add_textbox(Inches(x_pos + 0.1), Inches(2.2), Inches(1.8), Inches(0.7))
        val_frame = val_text.text_frame
        val_frame.text = value
        val_para = val_frame.paragraphs[0]
        val_para.font.size = Pt(42)
        val_para.font.bold = True
        val_para.font.color.rgb = color2
        val_para.alignment = PP_ALIGN.CENTER
        
        # Label
        label_text = slide.shapes.add_textbox(Inches(x_pos + 0.1), Inches(3), Inches(1.8), Inches(0.7))
        label_frame = label_text.text_frame
        label_frame.text = label
        label_para = label_frame.paragraphs[0]
        label_para.font.size = Pt(15)
        label_para.font.color.rgb = SLATE_600
        label_para.alignment = PP_ALIGN.CENTER
        
        x_pos += 2.1
    
    # Insight box
    insight_box = create_rounded_rectangle(
        slide, Inches(1), Inches(4.7), Inches(8), Inches(1.1),
        GREEN_100, GREEN_500, 3
    )
    insight_text = insight_box.text_frame
    insight_text.text = "💡 Model successfully ranks future defaults in a rising-rate environment"
    insight_para = insight_text.paragraphs[0]
    insight_para.font.size = Pt(24)
    insight_para.font.bold = True
    insight_para.font.color.rgb = GREEN_600
    insight_para.alignment = PP_ALIGN.CENTER
    insight_text.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    # ========== SLIDE 9: SHAP INSIGHTS ==========
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_gradient_background(slide, SLATE_50, SLATE_100)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(9), Inches(0.7))
    title_frame = title_box.text_frame
    title_frame.text = "Key Risk Drivers (SHAP Analysis)"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(44)
    title_para.font.bold = True
    title_para.font.color.rgb = SLATE_800
    
    drivers = [
        (1, "Fed Funds Rate at Origination", "2022-23 vintages are riskiest", 3.17, 1.0),
        (2, "30-yr Mortgage Rate at Origination", "High-rate environment = high risk", 1.15, 0.36),
        (3, "Original Interest Rate", "Direct borrower cost impact", 0.85, 0.27),
        (4, "HPI YoY Change at Origination", "Negative momentum signals distress", 0.72, 0.23),
        (5, "Loan Age", "Seasoning effect", 0.68, 0.21),
        (6, "Credit Score", "Important but not dominant", 0.54, 0.17)
    ]
    
    y_pos = 1.25
    for rank, name, insight, impact, normalized in drivers:
        # Main box
        box = create_rounded_rectangle(
            slide, Inches(0.7), Inches(y_pos), Inches(8.6), Inches(0.7),
            WHITE
        )
        box.shadow.inherit = False
        
        # Rank circle
        rank_circle = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            Inches(0.9), Inches(y_pos + 0.1), Inches(0.5), Inches(0.5)
        )
        rank_circle.fill.solid()
        rank_circle.fill.fore_color.rgb = PURPLE_600 if rank <= 3 else PURPLE_500
        rank_circle.line.fill.background()
        
        rank_text = rank_circle.text_frame
        rank_text.text = str(rank)
        rank_para = rank_text.paragraphs[0]
        rank_para.font.size = Pt(22)
        rank_para.font.bold = True
        rank_para.font.color.rgb = WHITE
        rank_para.alignment = PP_ALIGN.CENTER
        rank_text.vertical_anchor = MSO_ANCHOR.MIDDLE
        
        # Driver name and insight
        text_box = slide.shapes.add_textbox(Inches(1.6), Inches(y_pos + 0.08), Inches(5.5), Inches(0.54))
        text_frame = text_box.text_frame
        text_frame.text = f"{name}  •  {insight}"
        text_para = text_frame.paragraphs[0]
        text_para.font.size = Pt(13)
        text_para.font.bold = True if rank <= 3 else False
        text_para.font.color.rgb = SLATE_800
        
        # Impact bar background
        bar_bg = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(7.3), Inches(y_pos + 0.2), Inches(1.5), Inches(0.3)
        )
        bar_bg.fill.solid()
        bar_bg.fill.fore_color.rgb = SLATE_200
        bar_bg.line.fill.background()
        
        # Impact bar fill
        bar_width = 1.5 * normalized
        if bar_width > 0.1:
            bar_fill = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(7.3), Inches(y_pos + 0.2), Inches(bar_width), Inches(0.3)
            )
            bar_fill.fill.solid()
            bar_fill.fill.fore_color.rgb = PURPLE_500
            bar_fill.line.fill.background()
        
        # Impact value
        impact_text = slide.shapes.add_textbox(Inches(8.9), Inches(y_pos + 0.15), Inches(0.3), Inches(0.4))
        impact_frame = impact_text.text_frame
        impact_frame.text = f"{impact:.2f}"
        impact_para = impact_frame.paragraphs[0]
        impact_para.font.size = Pt(11)
        impact_para.font.color.rgb = SLATE_600
        
        y_pos += 0.8
    
    # Key finding box
    finding_box = create_rounded_rectangle(
        slide, Inches(1), Inches(6.3), Inches(8), Inches(0.85),
        ORANGE_100, ORANGE_500, 3
    )
    finding_text = finding_box.text_frame
    finding_text.text = "💡 Macro conditions at origination matter MORE than borrower credit score"
    finding_para = finding_text.paragraphs[0]
    finding_para.font.size = Pt(22)
    finding_para.font.bold = True
    finding_para.font.color.rgb = ORANGE_600
    finding_para.alignment = PP_ALIGN.CENTER
    finding_text.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    # ========== SLIDE 10: ECONOMIC INSIGHTS ==========
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_gradient_background(slide, SLATE_50, SLATE_100)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(9), Inches(0.7))
    title_frame = title_box.text_frame
    title_frame.text = "Economic Insights"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(48)
    title_para.font.bold = True
    title_para.font.color.rgb = SLATE_800
    
    insights = [
        ("📈", "2022-2023 vintages show significantly higher risk due to rate environment"),
        ("🌍", "Origination macro conditions dominate over borrower characteristics"),
        ("🏠", "Negative house price momentum is a strong default predictor"),
        ("⚖️", "Judicial foreclosure states show moderately elevated risk"),
        ("💳", "Credit score matters, but ranks 6th among all drivers")
    ]
    
    y_pos = 1.8
    for emoji, text in insights:
        box = create_rounded_rectangle(
            slide, Inches(1), Inches(y_pos), Inches(8), Inches(0.75),
            WHITE
        )
        box.shadow.inherit = False
        
        # Emoji
        emoji_text = slide.shapes.add_textbox(Inches(1.3), Inches(y_pos + 0.1), Inches(0.5), Inches(0.55))
        emoji_frame = emoji_text.text_frame
        emoji_frame.text = emoji
        emoji_para = emoji_frame.paragraphs[0]
        emoji_para.font.size = Pt(32)
        emoji_para.alignment = PP_ALIGN.CENTER
        emoji_text.vertical_anchor = MSO_ANCHOR.MIDDLE
        
        # Text
        text_box = slide.shapes.add_textbox(Inches(2), Inches(y_pos + 0.1), Inches(6.7), Inches(0.55))
        text_frame = text_box.text_frame
        text_frame.text = text
        text_para = text_frame.paragraphs[0]
        text_para.font.size = Pt(18)
        text_para.font.color.rgb = SLATE_700
        text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        
        y_pos += 0.95
    
    # ========== SLIDE 11: LIMITATIONS & FUTURE ==========
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_gradient_background(slide, SLATE_50, SLATE_100)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(9), Inches(0.7))
    title_frame = title_box.text_frame
    title_frame.text = "Limitations & Future Enhancements"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(42)
    title_para.font.bold = True
    title_para.font.color.rgb = SLATE_800
    
    # Limitations box
    lim_box = create_rounded_rectangle(
        slide, Inches(0.5), Inches(1.5), Inches(4.5), Inches(4.7),
        WHITE, RED_500, 4
    )
    
    # Header with background
    lim_header = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.7), Inches(1.7), Inches(4.1), Inches(0.6)
    )
    lim_header.fill.solid()
    lim_header.fill.fore_color.rgb = RED_500
    lim_header.line.fill.background()
    
    lim_header_text = lim_header.text_frame
    lim_header_text.text = "⚠️ Current Limitations"
    lim_header_para = lim_header_text.paragraphs[0]
    lim_header_para.font.size = Pt(22)
    lim_header_para.font.bold = True
    lim_header_para.font.color.rgb = WHITE
    lim_header_para.alignment = PP_ALIGN.CENTER
    lim_header_text.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    # Limitations list
    lim_text = slide.shapes.add_textbox(Inches(0.8), Inches(2.6), Inches(3.9), Inches(3.3))
    lim_frame = lim_text.text_frame
    limitations = [
        "Current macro variables not incorporated (only origination)",
        "No competing risk modeling with prepayment",
        "Missing delinquency burnout and payment shock features"
    ]
    lim_frame.text = "\n\n".join([f"• {item}" for item in limitations])
    for para in lim_frame.paragraphs:
        para.font.size = Pt(16)
        para.font.color.rgb = SLATE_700
        para.space_after = Pt(12)
    
    # Future box
    fut_box = create_rounded_rectangle(
        slide, Inches(5.2), Inches(1.5), Inches(4.5), Inches(4.7),
        WHITE, GREEN_500, 4
    )
    
    # Header with background
    fut_header = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(5.4), Inches(1.7), Inches(4.1), Inches(0.6)
    )
    fut_header.fill.solid()
    fut_header.fill.fore_color.rgb = GREEN_500
    fut_header.line.fill.background()
    
    fut_header_text = fut_header.text_frame
    fut_header_text.text = "✓ Future Enhancements"
    fut_header_para = fut_header_text.paragraphs[0]
    fut_header_para.font.size = Pt(22)
    fut_header_para.font.bold = True
    fut_header_para.font.color.rgb = WHITE
    fut_header_para.alignment = PP_ALIGN.CENTER
    fut_header_text.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    # Future list
    fut_text = slide.shapes.add_textbox(Inches(5.5), Inches(2.6), Inches(3.9), Inches(3.3))
    fut_frame = fut_text.text_frame
    enhancements = [
        "Add dynamic unemployment and HPI changes",
        "Implement LightGBM/XGBoost for higher AUC",
        "Multi-period survival or cure modeling",
        "Incorporate prepayment as competing risk"
    ]
    fut_frame.text = "\n\n".join([f"• {item}" for item in enhancements])
    for para in fut_frame.paragraphs:
        para.font.size = Pt(16)
        para.font.color.rgb = SLATE_700
        para.space_after = Pt(12)
    
    # ========== SLIDE 12: SUMMARY ==========
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_gradient_background(slide, SLATE_50, SLATE_100)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(9), Inches(0.7))
    title_frame = title_box.text_frame
    title_frame.text = "Key Takeaways"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(48)
    title_para.font.bold = True
    title_para.font.color.rgb = SLATE_800
    
    achievements = [
        "Built transparent default model on 20M+ real loan-month records",
        "Strict temporal split ensures credible out-of-time performance",
        "Macro origination conditions (Fed Funds & mortgage rates) dominate risk",
        "Production-ready: pickled pipeline + 95th percentile threshold",
        "End-to-end data science: engineering → modeling → validation → insights"
    ]
    
    y_pos = 1.7
    for i, achievement in enumerate(achievements, 1):
        box = create_rounded_rectangle(
            slide, Inches(1), Inches(y_pos), Inches(8), Inches(0.8),
            BLUE_600
        )
        box.shadow.inherit = False
        
        # Number circle
        num_circle = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            Inches(1.3), Inches(y_pos + 0.15), Inches(0.5), Inches(0.5)
        )
        num_circle.fill.solid()
        num_circle.fill.fore_color.rgb = WHITE
        num_circle.fill.transparency = 0.2
        num_circle.line.fill.background()
        
        num_text = num_circle.text_frame
        num_text.text = str(i)
        num_para = num_text.paragraphs[0]
        num_para.font.size = Pt(24)
        num_para.font.bold = True
        num_para.font.color.rgb = WHITE
        num_para.alignment = PP_ALIGN.CENTER
        num_text.vertical_anchor = MSO_ANCHOR.MIDDLE
        
        # Achievement text
        text_box = slide.shapes.add_textbox(Inches(2), Inches(y_pos + 0.15), Inches(6.7), Inches(0.5))
        text_frame = text_box.text_frame
        text_frame.text = achievement
        text_para = text_frame.paragraphs[0]
        text_para.font.size = Pt(18)
        text_para.font.bold = True
        text_para.font.color.rgb = WHITE
        text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        
        y_pos += 0.95
    
    # ========== SLIDE 13: THANK YOU ==========
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_gradient_background(slide, BLUE_900, BLUE_800)
    
    # Thank you text
    thanks_box = slide.shapes.add_textbox(Inches(2), Inches(2.2), Inches(6), Inches(1))
    thanks_frame = thanks_box.text_frame
    thanks_frame.text = "Thank You"
    thanks_para = thanks_frame.paragraphs[0]
    thanks_para.font.size = Pt(68)
    thanks_para.font.bold = True
    thanks_para.font.color.rgb = WHITE
    thanks_para.alignment = PP_ALIGN.CENTER
    
    # Divider
    divider = slide.shapes.add_connector(1, Inches(3.5), Inches(3.5), Inches(6.5), Inches(3.5))
    divider.line.color.rgb = BLUE_600
    divider.line.width = Pt(2)
    divider.line.transparency = 0.5
    
    # Contact info
    contact_box = slide.shapes.add_textbox(Inches(2), Inches(4), Inches(6), Inches(2))
    contact_frame = contact_box.text_frame
    contact_frame.text = "Divija Rao Balasankula\n\ndrbalasa@ncsu.edu\n\nQuestions?"
    
    contact_frame.paragraphs[0].font.size = Pt(28)
    contact_frame.paragraphs[0].font.color.rgb = BLUE_200
    contact_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    
    contact_frame.paragraphs[2].font.size = Pt(24)
    contact_frame.paragraphs[2].font.color.rgb = BLUE_300
    contact_frame.paragraphs[2].alignment = PP_ALIGN.CENTER
    
    contact_frame.paragraphs[4].font.size = Pt(32)
    contact_frame.paragraphs[4].font.color.rgb = BLUE_200
    contact_frame.paragraphs[4].alignment = PP_ALIGN.CENTER
    
    for para in contact_frame.paragraphs:
        para.space_after = Pt(10)
    
    # Save presentation
    prs.save('Mortgage_Default_Risk_Modeling.pptx')
    print("=" * 60)
    print("✓ Presentation created successfully!")
    print("=" * 60)
    print(f"📄 File: Mortgage_Default_Risk_Modeling.pptx")
    print(f"📊 Total slides: {len(prs.slides)}")
    print("=" * 60)

if __name__ == "__main__":
    create_presentation()