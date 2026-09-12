import re
import sqlite3
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO
from reportlab.lib import colors

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
import pymupdf


st.set_page_config(
    page_title="Legal Metrology Compliance Checker",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def load_easyocr_reader():
    """
    Load EasyOCR only once.
    """
    return easyocr.Reader(
        ["en"],
        gpu=True,
        verbose=False
    )


_easyocr_reader = None


def get_easyocr_reader():
    global _easyocr_reader

    if _easyocr_reader is None:
        _easyocr_reader = load_easyocr_reader()

    return _easyocr_reader


DB_NAME = "database.db"


def init_database():

    conn = sqlite3.connect(
        DB_NAME,
        check_same_thread=False
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            timestamp TEXT,
            score INTEGER,
            status TEXT,
            ocr_confidence REAL,
            extracted_text TEXT
        )
    """)

    conn.commit()

    cursor.execute("PRAGMA table_info(scans)")

    existing_columns = {
        row[1]
        for row in cursor.fetchall()
    }

    required_columns = {
        "filename": "TEXT",
        "timestamp": "TEXT",
        "score": "INTEGER",
        "status": "TEXT",
        "ocr_confidence": "REAL",
        "extracted_text": "TEXT",
    }

    for column_name, column_type in required_columns.items():

        if column_name not in existing_columns:

            cursor.execute(
                f"ALTER TABLE scans ADD COLUMN {column_name} {column_type}"
            )

    conn.commit()

    return conn


conn = init_database()


def pdf_to_images(pdf_bytes):
    """
    Convert every PDF page into a PNG image.

    Each page is returned as:
        (page_number, image_bytes)

    so it can be processed by the existing OCR pipeline
    exactly like a normal uploaded image.
    """

    pdf_document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    try:

        for page_index in range(
            len(pdf_document)
        ):

            page = pdf_document.load_page(
                page_index
            )

            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                alpha=False
            )

            image_bytes = pixmap.tobytes(
                "png"
            )

            pages.append(
                (
                    page_index + 1,
                    image_bytes
                )
            )

    finally:

        pdf_document.close()

    return pages


def bytes_to_image(image_bytes):

    if hasattr(
        image_bytes,
        "getvalue"
    ):

        raw_bytes = image_bytes.getvalue()

    else:

        raw_bytes = bytes(
            image_bytes
        )

    file_bytes = np.frombuffer(
        raw_bytes,
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise ValueError(
            "Unable to decode image."
        )

    return img


def upscale_image(
    img,
    max_dim=1400,
    min_dim=900
):
    """
    Resize down when the image is larger than max_dim, or up
    when it's smaller than min_dim.

    Keeping max_dim around 1400 instead of 2000 helps
    significantly with OCR speed. Small/far-away label photos
    are upscaled back up toward min_dim (capped by max_dim) so
    small print doesn't collapse to only a couple of pixels
    tall, which is a common cause of missed detections.
    """

    h, w = img.shape[:2]

    longest = max(
        h,
        w
    )

    shortest = min(
        h,
        w
    )

    if longest > max_dim:

        scale = (
            max_dim /
            longest
        )

        return cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    if shortest < min_dim and longest > 0:

        scale = min(
            min_dim / shortest,
            max_dim / longest
        )

        if scale > 1.0:

            return cv2.resize(
                img,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

    return img


def preprocess_fallback(img):
   
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Improve local contrast

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    contrast = clahe.apply(
        gray
    )

    # Mild denoising

    contrast = cv2.GaussianBlur(
        contrast,
        (3, 3),
        0
    )

    # Adaptive threshold

    adaptive = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return adaptive


# Below this variance (Laplacian of a grayscale, resolution-normalized
# image), a capture is treated as blurry enough to need dedicated
# handling rather than the standard fallback pass.
BLUR_VARIANCE_THRESHOLD = 60.0


def compute_blur_score(img):
    """
    Variance of the Laplacian as a focus/sharpness measure.

    Lower values mean a softer, more out-of-focus image. This is
    computed on the already-resized working image so the score is
    comparable across captures of different original resolutions.
    """

    gray = (
        cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )
        if img.ndim == 3
        else img
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return float(
        laplacian.var()
    )


def is_blurry(
    img,
    threshold=BLUR_VARIANCE_THRESHOLD
):

    return (
        compute_blur_score(img) <
        threshold
    )


def sharpen_image(
    img,
    amount=1.0,
    radius=3
):
    """
    Unsharp mask: subtract a blurred copy from the original to
    push back contrast lost to focus blur or motion blur, without
    the artifacts a naive sharpening kernel introduces.
    """

    blurred = cv2.GaussianBlur(
        img,
        (0, 0),
        radius
    )

    sharpened = cv2.addWeighted(
        img,
        1 + amount,
        blurred,
        -amount,
        0
    )

    return sharpened


def preprocess_blur_fallback(img):
    """
    Preprocessing variant tailored to blurry captures.

    Runs an unsharp mask first to recover edge contrast, then a
    slightly stronger CLAHE pass, then an edge-preserving
    (bilateral) denoise instead of the Gaussian blur used in
    preprocess_fallback, since a Gaussian blur would immediately
    undo the sharpening step on an already-soft image.
    """

    sharpened = sharpen_image(
        img,
        amount=1.3,
        radius=4
    )

    gray = cv2.cvtColor(
        sharpened,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    contrast = clahe.apply(
        gray
    )

    contrast = cv2.bilateralFilter(
        contrast,
        d=5,
        sigmaColor=40,
        sigmaSpace=40
    )

    adaptive = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return adaptive


def normalize_text(text):

    text = str(
        text
    ).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def normalize_ocr_text(text):

    text = normalize_text(
        text
    )

    replacements = {

        "NFD": "MFD",
        "MFO": "MFD",
        "MFC": "MFD",
        "NFO": "MFD",
        "MFD.": "MFD",
        "MFG.": "MFG",
        "R5.": "RS.",
        "R5": "RS",
        "R S": "RS",
        "INCL0F": "INCLUSIVE OF",
        "INCL OF": "INCLUSIVE OF",
        "INCL. OF": "INCLUSIVE OF",
        "INCLUS1VE": "INCLUSIVE",
        "INCLUSlVE": "INCLUSIVE",

    }

    upper = text.upper()

    for wrong, right in replacements.items():
        upper = upper.replace(
            wrong,
            right
        )

    return upper


def fuzzy_match(
    text,
    target,
    threshold=0.70
):

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper()
    )

    target = re.sub(
        r"[^A-Z0-9]",
        "",
        str(target).upper()
    )

    if not text or not target:
        return False

    if text == target:
        return True

    return SequenceMatcher(
        None,
        text,
        target
    ).ratio() >= threshold


def run_easyocr(image):

    reader = get_easyocr_reader()

    return reader.readtext(
        image,
        detail=1,
        paragraph=False,

        mag_ratio=1.2,
        text_threshold=0.35,
        low_text=0.25,

        link_threshold=0.3,

        rotation_info=None,
    )


def bbox_to_rect(bbox):

    xs = [
        float(point[0])
        for point in bbox
    ]

    ys = [
        float(point[1])
        for point in bbox
    ]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys)
    )


def iou(
    rect1,
    rect2
):

    x1 = max(
        rect1[0],
        rect2[0]
    )

    y1 = max(
        rect1[1],
        rect2[1]
    )

    x2 = min(
        rect1[2],
        rect2[2]
    )

    y2 = min(
        rect1[3],
        rect2[3]
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):
        return 0.0

    intersection = (
        (x2 - x1) *
        (y2 - y1)
    )

    area1 = (
        (rect1[2] - rect1[0]) *
        (rect1[3] - rect1[1])
    )

    area2 = (
        (rect2[2] - rect2[0]) *
        (rect2[3] - rect2[1])
    )

    denominator = (
        area1 +
        area2 -
        intersection
    )

    if denominator <= 0:
        return 0.0

    return (
        intersection /
        denominator
    )


def merge_detections(
    all_results,
    iou_threshold=0.45
):

    normalized = []

    for (
        bbox,
        text,
        confidence,
        engine,
        variant
    ) in all_results:

        text = normalize_text(
            text
        )

        if not text:
            continue

        confidence = float(
            confidence
        )

        normalized.append(
            (
                bbox,
                text,
                confidence,
                engine,
                variant
            )
        )

    normalized.sort(
        key=lambda item: item[2],
        reverse=True
    )

    kept = []
    kept_rects = []

    for item in normalized:

        (
            bbox,
            text,
            confidence,
            engine,
            variant
        ) = item

        rect = bbox_to_rect(
            bbox
        )

        duplicate = False

        for kept_rect in kept_rects:

            if iou(
                rect,
                kept_rect
            ) > iou_threshold:

                duplicate = True
                break

        if duplicate:
            continue

        kept.append(
            item
        )

        kept_rects.append(
            rect
        )

    return kept


def order_by_layout(results):

    items = []

    for (
        bbox,
        text,
        confidence,
        engine,
        variant
    ) in results:

        xs = [
            float(point[0])
            for point in bbox
        ]

        ys = [
            float(point[1])
            for point in bbox
        ]

        items.append(
            (
                min(ys),
                min(xs),
                bbox,
                text,
                confidence,
                engine,
                variant
            )
        )

    if not items:
        return []

    items.sort(
        key=lambda item: (
            item[0],
            item[1]
        )
    )

    return [
        (
            bbox,
            text,
            confidence,
            engine,
            variant
        )
        for (
            y,
            x,
            bbox,
            text,
            confidence,
            engine,
            variant
        ) in items
    ]


def calculate_ocr_confidence(
    results
):

    if not results:
        return 0.0

    values = [
        float(item[2])
        for item in results
    ]

    return (
        sum(values) /
        len(values)
    )


def extract_single_label_data(
    image_bytes,
    conf_threshold=0.30
):

    original = bytes_to_image(
        image_bytes
    )

    # Smaller max dimension = considerably faster OCR.
    # Small/far-away captures also get upscaled back up inside
    # upscale_image so tiny print doesn't fall below what OCR
    # can resolve.

    img = upscale_image(
        original,
        max_dim=1400
    )

    blur_score = compute_blur_score(
        img
    )

    blurry = blur_score < BLUR_VARIANCE_THRESHOLD

    all_results = []

    # For a blurry capture, feed EasyOCR an unsharp-masked version
    # on the primary pass instead of the raw soft image — sharp
    # edges recover a meaningful share of detections that would
    # otherwise only show up on the fallback pass.

    primary_source = (
        sharpen_image(
            img,
            amount=1.0,
            radius=3
        )
        if blurry
        else img
    )

    primary_variant = (
        "sharpened"
        if blurry
        else "original"
    )

    try:

        raw_results = run_easyocr(
            primary_source
        )

        for result in raw_results:

            if (
                not isinstance(
                    result,
                    (tuple, list)
                )
                or
                len(result) != 3
            ):
                continue

            bbox, text, confidence = result

            try:

                confidence = float(
                    confidence
                )

            except Exception:
                continue

            if confidence >= conf_threshold:

                all_results.append(
                    (
                        bbox,
                        text,
                        confidence,
                        "easyocr",
                        primary_variant
                    )
                )

    except Exception:
        pass


    first_count = len(
        all_results
    )

    first_confidence = (
        calculate_ocr_confidence(
            all_results
        )
        if all_results
        else 0.0
    )

    needs_fallback = (
        first_count < 3
        or
        first_confidence < 0.55
        or
        blurry
    )

    if needs_fallback:

        fallback_variants = []

        if blurry:

            fallback_variants.append(
                (
                    "blur_fallback",
                    preprocess_blur_fallback(img)
                )
            )

        fallback_variants.append(
            (
                "adaptive_fallback",
                preprocess_fallback(img)
            )
        )

        for variant_name, variant_image in fallback_variants:

            try:

                raw_results = run_easyocr(
                    variant_image
                )

                for result in raw_results:

                    if (
                        not isinstance(
                            result,
                            (tuple, list)
                        )
                        or
                        len(result) != 3
                    ):
                        continue

                    bbox, text, confidence = result

                    try:

                        confidence = float(
                            confidence
                        )

                    except Exception:
                        continue

                    if confidence >= conf_threshold:

                        all_results.append(
                            (
                                bbox,
                                text,
                                confidence,
                                "easyocr",
                                variant_name
                            )
                        )

            except Exception:
                pass

    merged = merge_detections(
        all_results,
        iou_threshold=0.45
    )

    ordered = order_by_layout(
        merged
    )

    extracted_text = "\n".join(
        item[1]
        for item in ordered
    )

    confidence = calculate_ocr_confidence(
        ordered
    )

    return (
        extracted_text,
        ordered,
        confidence,
        original
    )


def get_confidence_for_keywords(
    all_image_results,
    keywords
):

    keyword_confidences = []

    for img_data in all_image_results:

        for (
            bbox,
            text,
            confidence,
            engine,
            variant
        ) in img_data["ocr_results"]:

            upper_text = normalize_ocr_text(
                text
            )

            for keyword in keywords:

                if (
                    keyword.upper()
                    in upper_text
                ):

                    keyword_confidences.append(
                        confidence
                    )

                    break

    if not keyword_confidences:
        return None

    return (
        sum(keyword_confidences) /
        len(keyword_confidences)
    )


def contains_fuzzy_keyword(
    text,
    keywords,
    threshold=0.72
):

    words = re.findall(
        r"[A-Z0-9]+",
        text.upper()
    )

    for word in words:

        for keyword in keywords:

            if fuzzy_match(
                word,
                keyword,
                threshold
            ):

                return True

    return False


def evaluate_compliance(
    aggregated_text,
    all_image_results
):

    text_upper = normalize_ocr_text(
        aggregated_text
    )

    # Replace separators

    cleaned_text = re.sub(
        r"[;:|]",
        ".",
        text_upper
    )

    checks = {}
    field_confidence = {}

    mrp_patterns = [

        # MRP 425

        r"\bM\.?\s*R\.?\s*P\.?\s*[:\-]?\s*₹?\s*\d{1,5}(?:\.\d{1,2})?",
        r"\bM\.?\s*R\.?\s*P\.?\s*[:\-]?\s*(?:RS\.?|R5\.?)?\s*₹?\s*\d{1,5}(?:\.\d{1,2})?",
        r"\bRS\.?\s*₹?\s*\d{1,5}(?:\.\d{1,2})?",
        r"₹\s*\d{1,5}(?:\.\d{1,2})?",
        r"\b\d{1,5}\.\d{2}\b",

    ]

    mrp_found = any(
        re.search(
            pattern,
            text_upper
        )
        for pattern in mrp_patterns
    )

    if not mrp_found:

        mrp_found = (
            contains_fuzzy_keyword(
                text_upper,
                ["MRP"],
                threshold=0.65
            )
            and
            bool(
                re.search(
                    r"\d{1,5}",
                    text_upper
                )
            )
        )

    checks["MRP Present"] = mrp_found

    field_confidence[
        "MRP Present"
    ] = get_confidence_for_keywords(
        all_image_results,
        [
            "MRP",
            "RS",
            "R5",
            "425",
            "10.00",
            "153"
        ]
    )
  
    inclusive_patterns = [

        r"INCLUSIVE.{0,20}OF.{0,20}ALL.{0,20}TAX",
        r"INCLUSIVE.{0,20}OF.{0,20}TAX",
        r"INCL.{0,15}OF.{0,15}ALL.{0,15}TAX",
        r"INCL.{0,15}TAX",
        r"INCLUSIVE.{0,30}TAX",

    ]

    inclusive_found = any(
        re.search(
            pattern,
            text_upper
        )
        for pattern in inclusive_patterns
    )

    checks[
        "MRP Inclusive Taxes Mentioned"
    ] = inclusive_found

    field_confidence[
        "MRP Inclusive Taxes Mentioned"
    ] = get_confidence_for_keywords(
        all_image_results,
        [
            "TAX",
            "INCL",
            "INCLUSIVE",
            "ALL"
        ]
    )

    quantity_patterns = [

        r"\bNET\s+(?:WEIGHT|WT|QTY|QUANTITY)\s*[:\-]?\s*"
        r"\d+(?:\.\d+)?\s*"
        r"(?:G|GM|GMS|GRAM|GRAMS|KG|KGS|ML|L|LTR|"
        r"LITRE|LITRES|PCS|PC|UNITS)\b",

        r"\b\d+(?:\.\d+)?\s*"
        r"(?:G|GM|GMS|GRAM|GRAMS|KG|KGS|ML|L|LTR|"
        r"LITRE|LITRES|PCS|PC|UNITS)\b",

        r"\b\d+(?:\.\d+)?(?:G|GM|GMS|KG|ML|LTR)\b",

    ]

    quantity_found = any(
        re.search(
            pattern,
            text_upper
        )
        for pattern in quantity_patterns
    )

    checks[
        "Net Quantity Present"
    ] = quantity_found

    field_confidence[
        "Net Quantity Present"
    ] = get_confidence_for_keywords(
        all_image_results,
        [
            "NET",
            "WEIGHT",
            "QUANTITY",
            "QTY",
            "G",
            "GM",
            "KG",
            "ML"
        ]
    )

    date_patterns = [

        r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",
        r"\b\d{1,2}[/.-]\d{2,4}\b",
        r"\b\d{1,2}\s*[-/]\s*\d{2}\b",

        r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|"
        r"OCT|NOV|DEC)[A-Z]?\s*[-/]?\s*\d{2,4}\b",

    ]

    has_date_value = any(
        re.search(
            pattern,
            text_upper
        )
        for pattern in date_patterns
    )

    date_keyword_found = (
        bool(
            re.search(
                r"\b(?:MFG|MFD|MFO|NFD|PKD|"
                r"PACKED|MANUFACTURED|"
                r"MANUFACTURING|DATE|EXP)\b",
                text_upper
            )
        )
        or
        contains_fuzzy_keyword(
            text_upper,
            [
                "MFD",
                "MFG",
                "PKD"
            ],
            threshold=0.62
        )
    )

    date_found = (
        has_date_value
        and
        date_keyword_found
    )

    if not date_found:

        date_found = bool(
            re.search(
                r"(?:MFD|MFG|NFD|PKD|PACKED|"
                r"MANUFACTURED)"
                r".{0,40}"
                r"(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|"
                r"\d{1,2}[/.-]\d{2,4})",
                text_upper
            )
        )

    checks[
        "Mfg/Packing Date Present"
    ] = date_found

    field_confidence[
        "Mfg/Packing Date Present"
    ] = get_confidence_for_keywords(
        all_image_results,
        [
            "MFG",
            "MFD",
            "NFD",
            "MFO",
            "PKD",
            "PACKED",
            "MANUFACTURED",
            "DATE",
            "EXP"
        ]
    )


    origin_phrases = [

        "MADE IN",
        "COUNTRY OF ORIGIN",
        "PRODUCT OF INDIA",
        "ORIGIN INDIA",
        "ORIGIN: INDIA",
        "MANUFACTURED IN INDIA",
        "MADE IN INDIA",
        "COUNTRY INDIA",

        "INDIA",
        "KARNATAKA",
        "BENGALURU",
        "BANGALORE",
        "WHITEFIELD",
        "MUMBAI",
        "GUWAHATI",
        "ASSAM",

        "MANUFACTURED BY",
        "MANUFACTURED IN",
        "MFD BY",
        "PKD BY"
        "MARKETED BY",

    ]

    origin_found = any(
        phrase in text_upper
        for phrase in origin_phrases
    )

    checks[
        "Country of Origin Present"
    ] = origin_found

    field_confidence[
        "Country of Origin Present"
    ] = get_confidence_for_keywords(
        all_image_results,
        [
            "MADE",
            "COUNTRY",
            "ORIGIN",
            "INDIA",
            "KARNATAKA",
            "BENGALURU",
            "WHITEFIELD",
            "MUMBAI"
        ]
    )

    customer_care_keywords = [

        "CUSTOMER CARE",
        "CONSUMER CARE",
        "TOLL FREE",
        "HELPLINE",
        "EMAIL",
        "CONTACT US",
        "CALL US",
        "EXECUTIVE",
        "CONSUMER",
        "REGD OFFICE",
        "REGISTERED OFFICE",
        "ADDRESS",
        "PHONE",
        "PH",
        "TEL",
        "@",

    ]

    customer_found = any(
        keyword in text_upper
        for keyword in customer_care_keywords
    )

    phone_found = bool(
        re.search(
            r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
            text_upper
        )
    )

    email_found = bool(
        re.search(
            r"\b[A-Z0-9.\_%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            text_upper,
            re.IGNORECASE
        )
    )

    customer_found = (
        customer_found
        or
        phone_found
        or
        email_found
    )

    checks[
        "Customer Care Details"
    ] = customer_found

    field_confidence[
        "Customer Care Details"
    ] = get_confidence_for_keywords(
        all_image_results,
        customer_care_keywords
    )
  
    total_passed = sum(
        1
        for value in checks.values()
        if value
    )

    total_checks = len(
        checks
    )

    score = int(
        (
            total_passed /
            total_checks
        ) * 100
    )

    status = (
        "COMPLIANT"
        if score >= 100
        else
        "NON-COMPLIANT"
    )

    return (
        checks,
        field_confidence,
        score,
        status
    )


def format_ocr_results(results):

    rows = []

    for (
        bbox,
        text,
        confidence,
        engine,
        variant
    ) in results:

        xs = [
            int(point[0])
            for point in bbox
        ]

        ys = [
            int(point[1])
            for point in bbox
        ]

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)

        rows.append({

            "Text": text,

            "Confidence":
                round(
                    confidence * 100,
                    2
                ),

            "Engine": engine,

            "Preprocessing": variant,

            "X": x1,

            "Y": y1,

            "Width":
                x2 - x1,

            "Height":
                y2 - y1,

        })

    return pd.DataFrame(
        rows
    )


def create_ocr_overlay(
    image,
    results
):

    output = image.copy()

    for (
        bbox,
        text,
        confidence,
        engine,
        variant
    ) in results:

        points = np.asarray(
            bbox,
            dtype=np.int32
        )

        cv2.polylines(
            output,
            [points],
            True,
            (0, 255, 0),
            2
        )

        x = int(
            min(
                point[0]
                for point in bbox
            )
        )

        y = int(
            min(
                point[1]
                for point in bbox
            )
        )

        label = (
            f"{confidence * 100:.0f}%"
        )

        cv2.putText(
            output,
            label,
            (
                x,
                max(y - 5, 15)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return output


def save_scan(
    filename,
    score,
    status,
    ocr_confidence,
    extracted_text
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    conn.execute(
        """
        INSERT INTO scans
        (
            filename,
            timestamp,
            score,
            status,
            ocr_confidence,
            extracted_text
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            timestamp,
            score,
            status,
            round(
                ocr_confidence * 100,
                2
            ),
            extracted_text,
        )
    )

    conn.commit()


RULE_DISPLAY_NAMES = [

    "MRP Present",

    "MRP Inclusive Taxes Mentioned",

    "Net Quantity Present",

    "Mfg/Packing Date Present",

    "Country of Origin Present",

    "Customer Care Details",

]


def display_progressive_compliance(
    checks,
    field_confidence,
    finished=False
):

    st.subheader(
        "Mandatory Declaration Audit"
    )

    for rule in RULE_DISPLAY_NAMES:

        passed = checks.get(
            rule,
            False
        )

        confidence = field_confidence.get(
            rule
        )

        left, right = st.columns(
            [5, 1]
        )

        with left:

            if passed:

                st.success(
                    f"✓ {rule}"
                )

            elif finished:

                st.error(
                    f"✗ {rule} — Not detected"
                )

            else:

                st.info(
                    f"⏳ {rule} — Scanning..."
                )

        with right:

            if confidence is not None:

                st.write(
                    f"{confidence * 100:.1f}% OCR"
                )

            else:

                st.write(
                    "—"
                )



# ============================================================
# UI / APP PRESENTATION LAYER
# ============================================================

import json
import html
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)

# -----------------------------
# App-level state
# -----------------------------
if "ui_upload_cycle" not in st.session_state:
    st.session_state.ui_upload_cycle = 0
if "ui_last_scan_id" not in st.session_state:
    st.session_state.ui_last_scan_id = None
if "ui_last_report" not in st.session_state:
    st.session_state.ui_last_report = None
if "ui_manual_checks" not in st.session_state:
    st.session_state.ui_manual_checks = {}
if "ui_scan_active" not in st.session_state:
    st.session_state.ui_scan_active = False

CATEGORY_OPTIONS = [
    "Food & Beverage",
    "Cosmetic and Personal Care",
    "Medicine",
    "Electronics",
    "Household Goods",
    "Apparel and Textiles",
    "Others",
]

ACCENT = "#ff4b4b"
BG = "#080808"
PANEL = "#111111"
BORDER = "#242424"
MUTED = "#9a9a9a"
WHITE = "#f4f4f4"
GREEN = "#26c281"
AMBER = "#ffb020"

RULE_LABELS = list(RULE_DISPLAY_NAMES)

# -----------------------------
# Auxiliary metadata table
# -----------------------------
UI_META_TABLE = "inspection_metadata"


def init_ui_metadata_table():
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {UI_META_TABLE} (
            scan_id INTEGER PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            remark TEXT,
            final_score INTEGER,
            final_status TEXT,
            checks_json TEXT,
            manual_checks_json TEXT,
            created_at TEXT,
            FOREIGN KEY(scan_id) REFERENCES scans(id)
        )
        """
    )
    conn.commit()


init_ui_metadata_table()

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    f"""
    <style>
        :root {{
            --accent: {ACCENT};
            --bg: {BG};
            --panel: {PANEL};
            --border: {BORDER};
            --muted: {MUTED};
        }}

        .stApp {{
            background: var(--bg);
            color: {WHITE};
        }}

        [data-testid="stHeader"] {{
            background: rgba(8,8,8,0.90);
        }}

        [data-testid="stSidebar"] {{
            background: #0b0b0b;
        }}

        [data-testid="stFileUploader"] section {{
            min-height: 210px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px dashed #3a3a3a;
            border-radius: 18px;
            background: linear-gradient(145deg, #111111, #0b0b0b);
            transition: border-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
        }}

        [data-testid="stFileUploader"] section:hover {{
            border-color: var(--accent);
            transform: translateY(-1px);
            box-shadow: 0 0 0 1px rgba(255,75,75,0.16), 0 14px 32px rgba(0,0,0,0.28);
        }}

        [data-testid="stCameraInput"] {{
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}

        div.stButton > button,
        div.stDownloadButton > button {{
            border-radius: 11px;
            min-height: 44px;
            border: 1px solid #2d2d2d;
            transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
        }}

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {{
            transform: translateY(-1px);
            border-color: var(--accent);
            box-shadow: 0 8px 22px rgba(0,0,0,0.24);
        }}

        .hero {{
            padding: 24px 28px;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: radial-gradient(circle at 80% 20%, rgba(255,75,75,0.12), transparent 35%), #0f0f0f;
            margin-bottom: 20px;
        }}

        .hero-kicker {{
            color: var(--accent);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }}

        .hero-title {{
            margin: 0;
            font-size: clamp(30px, 4vw, 48px);
            line-height: 1.05;
            font-weight: 800;
        }}

        .hero-copy {{
            color: #a8a8a8;
            max-width: 760px;
            margin-top: 10px;
            font-size: 15px;
        }}

        .section-title {{
            font-size: 21px;
            font-weight: 750;
            margin: 5px 0 12px;
        }}

        .mini-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
            height: 100%;
        }}

        .mini-label {{
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .mini-value {{
            margin-top: 6px;
            font-size: 20px;
            font-weight: 700;
        }}

        .audit-shell {{
            background: #0d0d0d;
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 20px;
            margin-top: 18px;
        }}

        .audit-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 13px 14px;
            margin: 8px 0;
            border-radius: 12px;
            background: #121212;
            border: 1px solid #202020;
        }}

        .audit-icon {{
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 15px;
            font-weight: 800;
            flex: 0 0 28px;
        }}

        .audit-pass {{
            background: rgba(38,194,129,0.16);
            color: {GREEN};
        }}

        .audit-fail {{
            background: rgba(255,75,75,0.14);
            color: {ACCENT};
        }}

        .audit-wait {{
            background: #1b1b1b;
            color: #8c8c8c;
        }}

        .audit-manual {{
            background: rgba(255,176,32,0.14);
            color: {AMBER};
        }}

        .audit-text {{
            flex: 1;
            font-size: 14px;
            font-weight: 620;
        }}

        .audit-sub {{
            color: #888;
            font-size: 12px;
            margin-top: 2px;
        }}

        .status-pill {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.04em;
        }}

        .pill-pass {{ background: rgba(38,194,129,0.13); color: {GREEN}; }}
        .pill-fail {{ background: rgba(255,75,75,0.13); color: {ACCENT}; }}
        .pill-manual {{ background: rgba(255,176,32,0.13); color: {AMBER}; }}

        .remark-box {{
            background: #0f0f0f;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px 16px;
            color: #d6d6d6;
            margin-top: 8px;
        }}

        .db-table-note {{
            color: #858585;
            font-size: 12px;
            margin-top: -4px;
            margin-bottom: 12px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def reset_inspection():
    st.session_state.ui_upload_cycle += 1
    st.session_state.ui_last_scan_id = None
    st.session_state.ui_last_report = None
    st.session_state.ui_manual_checks = {}
    st.session_state.ui_scan_active = False
    for key in ["current_results", "current_name", "current_category", "current_remark"]:
        st.session_state.pop(key, None)


def build_scan_items(uploaded_files, camera_file):
    items = []

    for file in uploaded_files or []:
        name = getattr(file, "name", "uploaded-file")
        if name.lower().endswith(".pdf"):
            for page_number, page_bytes in pdf_to_images(file.getvalue()):
                items.append(
                    {
                        "filename": f"{name} — Page {page_number}",
                        "bytes": page_bytes,
                        "source_type": "pdf",
                        "page_number": page_number,
                    }
                )
        else:
            items.append(
                {
                    "filename": name,
                    "bytes": file.getvalue(),
                    "source_type": "image",
                    "page_number": None,
                }
            )

    if camera_file is not None:
        items.append(
            {
                "filename": "Camera Capture",
                "bytes": camera_file.getvalue(),
                "source_type": "camera",
                "page_number": None,
            }
        )

    return items


def criterion_low_conf(rule, field_confidence):
    confidence = field_confidence.get(rule)
    return confidence is None or float(confidence) < 0.50


def effective_checks(checks, field_confidence, manual_checks):
    final = {}
    manual_required = {}
    for rule in RULE_LABELS:
        raw_pass = bool(checks.get(rule, False))
        needs_manual = criterion_low_conf(rule, field_confidence)
        
        # Exception for MRP Present: Consider it compliant automatically but flag for manual check visually
        if rule == "MRP Present" and raw_pass:
            final[rule] = True
            manual_required[rule] = False  # Bypasses the hard check override
        else:
            manual_required[rule] = needs_manual
            final[rule] = raw_pass and not needs_manual
            if needs_manual and manual_checks.get(rule, False):
                final[rule] = True
    return final, manual_required


def audit_progress_html(checks, field_confidence, manual_checks=None, finished=False):
    manual_checks = manual_checks or {}
    rows = []
    passed_for_meter = 0

    for rule in RULE_LABELS:
        raw_pass = bool(checks.get(rule, False))
        needs_manual = criterion_low_conf(rule, field_confidence)
        manually_checked = bool(manual_checks.get(rule, False))

        is_mrp_exception = (rule == "MRP Present" and raw_pass and needs_manual)

        if is_mrp_exception:
            icon = "✓"
            icon_class = "audit-manual"
            sub = "Manual inspection recommended"
            passed_for_meter += 1
        elif finished:
            final_pass = (raw_pass and not needs_manual) or manually_checked
            if final_pass:
                icon = "✓"
                icon_class = "audit-pass"
                sub = "Verified"
                passed_for_meter += 1
            elif needs_manual:
                icon = "!"
                icon_class = "audit-manual"
                sub = "Manual inspection required"
            else:
                icon = "×"
                icon_class = "audit-fail"
                sub = "Not detected"
        else:
            if raw_pass and not needs_manual:
                icon = "✓"
                icon_class = "audit-pass"
                sub = "Verified"
                passed_for_meter += 1
            elif raw_pass and needs_manual:
                icon = "!"
                icon_class = "audit-manual"
                sub = "Low-confidence result"
            else:
                icon = "…"
                icon_class = "audit-wait"
                sub = "Checking"

        rows.append(
            f"""
            <div class=\"audit-row\">
                <div class=\"audit-icon {icon_class}\">{icon}</div>
                <div class=\"audit-text\">{html.escape(rule)}<div class=\"audit-sub\">{html.escape(sub)}</div></div>
            </div>
            """
        )

    meter = int((passed_for_meter / max(len(RULE_LABELS), 1)) * 100)
    return meter, "".join(rows)


def store_metadata(scan_id, product_name, category, remark, final_checks, final_score, final_status, manual_checks):
    conn.execute(
        f"""
        INSERT INTO {UI_META_TABLE}
            (scan_id, product_name, category, remark, final_score, final_status,
             checks_json, manual_checks_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(scan_id) DO UPDATE SET
            product_name = excluded.product_name,
            category = excluded.category,
            remark = excluded.remark,
            final_score = excluded.final_score,
            final_status = excluded.final_status,
            checks_json = excluded.checks_json,
            manual_checks_json = excluded.manual_checks_json,
            created_at = excluded.created_at
        """,
        (
            scan_id,
            product_name,
            category,
            remark,
            int(final_score),
            final_status,
            json.dumps(final_checks),
            json.dumps(manual_checks),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()


def load_history_dataframe():
    return pd.read_sql_query(
        f"""
        SELECT
            s.id,
            COALESCE(m.product_name, 'Unnamed Product') AS product_name,
            COALESCE(m.category, 'Others') AS category,
            s.filename,
            s.timestamp,
            COALESCE(m.final_score, s.score) AS score,
            COALESCE(m.final_status, s.status) AS status,
            COALESCE(m.remark, '') AS remark,
            s.ocr_confidence,
            COALESCE(m.checks_json, '{{}}') AS checks_json,
            COALESCE(m.manual_checks_json, '{{}}') AS manual_checks_json
        FROM scans s
        LEFT JOIN {UI_META_TABLE} m ON m.scan_id = s.id
        ORDER BY s.id DESC
        """,
        conn,
    )


def make_report_pdf(product_name, category, checks, manual_checks, remark, final_score, final_status, timestamp, field_confidence=None):
    # DejaVu has the tick/cross glyphs used by the report.
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", font_path))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", bold_path))
        body_font = "DejaVu"
        bold_font = "DejaVu-Bold"
    except Exception:
        body_font = "Helvetica"
        bold_font = "Helvetica-Bold"

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"Compliance Report — {product_name}",
        author="Legal Metrology Compliance Checker",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=22,
        leading=27,
        textColor=colors.HexColor("#111111"),
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    kicker_style = ParagraphStyle(
        "Kicker",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor(ACCENT),
        spaceAfter=6,
    )
    normal_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName=body_font,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#303030"),
    )
    small_style = ParagraphStyle(
        "ReportSmall",
        parent=normal_style,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#606060"),
    )
    cell_style = ParagraphStyle(
        "Cell",
        parent=normal_style,
        fontSize=9,
        leading=12,
    )
    cell_bold = ParagraphStyle(
        "CellBold",
        parent=cell_style,
        fontName=bold_font,
    )

    story = []
    story.append(Paragraph("LEGAL METROLOGY", kicker_style))
    story.append(Paragraph("Compliance Inspection Report", title_style))
    story.append(Paragraph("Packaging declaration audit", normal_style))
    story.append(Spacer(1, 9 * mm))

    status_color = "#1d8f63" if final_status == "COMPLIANT" else ACCENT
    meta_data = [
        [Paragraph("Product name", cell_bold), Paragraph(html.escape(product_name or "Unnamed Product"), cell_style)],
        [Paragraph("Category", cell_bold), Paragraph(html.escape(category or "Others"), cell_style)],
        [Paragraph("Compliance", cell_bold), Paragraph(f'<font color="{status_color}"><b>{final_status}</b></font> — {final_score}%', cell_style)],
        [Paragraph("Inspection time", cell_bold), Paragraph(html.escape(timestamp), cell_style)],
    ]
    meta_table = Table(meta_data, colWidths=[42 * mm, 132 * mm])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f5f5")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#dddddd")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e6e6e6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 10 * mm))

    story.append(Paragraph("Declaration checklist", ParagraphStyle("H", parent=normal_style, fontName=bold_font, fontSize=13, leading=17, spaceAfter=5)))

    checklist_rows = [[
        Paragraph("Status", cell_bold),
        Paragraph("Requirement", cell_bold),
        Paragraph("Inspection result", cell_bold),
    ]]
    for rule in RULE_LABELS:
        passed = bool(checks.get(rule, False))
        manually = bool(manual_checks.get(rule, False))
        if passed:
            icon = "✓"
            result = "Passed by OCR"
            if field_confidence and rule == "MRP Present" and criterion_low_conf(rule, field_confidence):
                result = "Passed by OCR (Manual inspection recommended)"
            if manually:
                result = "Passed — manual inspection"
        else:
            icon = "×"
            result = "Not detected"
            if manually:
                result = "Passed — manual inspection"
                icon = "✓"
        result_color = "#1d8f63" if icon == "✓" else ACCENT
        checklist_rows.append(
            [
                Paragraph(f'<font color="{result_color}"><b>{icon}</b></font>', cell_bold),
                Paragraph(html.escape(rule), cell_style),
                Paragraph(html.escape(result), cell_style),
            ]
        )

    check_table = Table(checklist_rows, colWidths=[18 * mm, 83 * mm, 73 * mm], repeatRows=1)
    check_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#151515")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#dcdcdc")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e7e7e7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(check_table)
    story.append(Spacer(1, 9 * mm))

    story.append(Paragraph("Inspector remark", ParagraphStyle("H2", parent=normal_style, fontName=bold_font, fontSize=12, leading=15, spaceAfter=4)))
    remark_text = html.escape(remark.strip()) if remark and remark.strip() else "No inspector remark provided."
    remark_table = Table([[Paragraph(remark_text, normal_style)]], colWidths=[174 * mm])
    remark_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafafa")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#dddddd")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(remark_table)
    story.append(Spacer(1, 9 * mm))
    story.append(Paragraph("This report reflects the compliance checks produced by the application and any manual inspection confirmations recorded during the inspection.", small_style))

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#dddddd"))
        canvas.line(18 * mm, 12 * mm, 192 * mm, 12 * mm)
        canvas.setFont(body_font, 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(18 * mm, 7 * mm, "Legal Metrology Compliance Checker")
        canvas.drawRightString(192 * mm, 7 * mm, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def get_current_result():
    return st.session_state.get("current_results")


def save_current_scan_as_final():
    data = get_current_result()
    if not data:
        return None

    checks = data["checks"]
    field_confidence = data["field_confidence"]
    manual_checks = st.session_state.ui_manual_checks
    final_checks, manual_required = effective_checks(checks, field_confidence, manual_checks)
    final_score = int(round((sum(final_checks.values()) / max(len(RULE_LABELS), 1)) * 100))
    final_status = "COMPLIANT" if final_score >= 80 else "NON-COMPLIANT"

    product_name = st.session_state.get("current_name", "").strip() or "Unnamed Product"
    category = st.session_state.get("current_category", "Others") or "Others"
    remark = st.session_state.get("current_remark", "").strip()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    scan_id = st.session_state.get("ui_last_scan_id")
    if scan_id is None:
        return None

    store_metadata(
        scan_id,
        product_name,
        category,
        remark,
        final_checks,
        final_score,
        final_status,
        manual_checks,
    )

    report_bytes = make_report_pdf(
        product_name,
        category,
        final_checks,
        manual_checks,
        remark,
        final_score,
        final_status,
        timestamp,
        field_confidence=field_confidence
    )
    st.session_state.ui_last_report = report_bytes

    return {
        "scan_id": scan_id,
        "checks": final_checks,
        "manual_required": manual_required,
        "manual_checks": dict(manual_checks),
        "score": final_score,
        "status": final_status,
        "name": product_name,
        "category": category,
        "remark": remark,
        "timestamp": timestamp,
    }


# -----------------------------
# Header / navigation
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">SIH26034 · Inspection Workspace</div>
        <div class="hero-title">Legal Metrology<br>Compliance Checker</div>
        <div class="hero-copy">Scan packaging from multiple sides, audit mandatory declarations live, and generate a clean inspection report.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_dashboard, tab_database = st.tabs(["◉ Dashboard", "▣ Database"])

# ============================================================
# DASHBOARD
# ============================================================
with tab_dashboard:
    if st.button("＋ New Inspection", type="primary", width='stretch', key="top_new_inspection"):
        reset_inspection()
        st.rerun()

    st.markdown('<div class="section-title">Inspection details</div>', unsafe_allow_html=True)
    info1, info2 = st.columns([1.5, 1], gap="large")
    with info1:
        product_name = st.text_input(
            "Product name",
            value=st.session_state.get("current_name", ""),
            placeholder="e.g. Organic Honey 500 g",
            key="current_name",
        )
    with info2:
        product_category = st.selectbox(
            "Category",
            CATEGORY_OPTIONS,
            index=CATEGORY_OPTIONS.index(st.session_state.get("current_category", CATEGORY_OPTIONS[0]))
            if st.session_state.get("current_category", CATEGORY_OPTIONS[0]) in CATEGORY_OPTIONS else 0,
            key="current_category",
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Add packaging images</div>', unsafe_allow_html=True)
    
    # Toggle functionality for File vs. Camera Uploads
    input_method = st.radio("Select input method:", ["Upload Files", "Camera Capture"], horizontal=True, label_visibility="collapsed")

    uploaded_files = None
    camera_file = None

    if input_method == "Upload Files":
        uploaded_files = st.file_uploader(
            "Drop images or PDF pages here",
            type=["jpg", "jpeg", "png", "webp", "pdf"],
            accept_multiple_files=True,
            key=f"media_upload_{st.session_state.ui_upload_cycle}",
        )
        st.caption("Images and PDFs are supported. Multiple packaging sides can be scanned together.")
    else:
        camera_file = st.camera_input(
            "Take a packaging photo",
            key=f"camera_upload_{st.session_state.ui_upload_cycle}",
        )
        st.caption("Use the camera for a live side capture.")

    scan_items = []
    input_error = None
    try:
        scan_items = build_scan_items(uploaded_files, camera_file)
    except Exception as exc:
        input_error = str(exc)

    if input_error:
        st.error("One of the selected PDF files could not be read. Please choose the file again.")

    if scan_items:
        st.markdown(
            f"<div class='mini-card'><div class='mini-label'>Ready to scan</div><div class='mini-value'>{len(scan_items)} image/page(s)</div></div>",
            unsafe_allow_html=True,
        )

        with st.expander("Preview selected sides", expanded=False):
            preview_cols = st.columns(min(4, len(scan_items)))
            for idx, item in enumerate(scan_items):
                try:
                    preview = bytes_to_image(item["bytes"])
                    with preview_cols[idx % len(preview_cols)]:
                        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), width='stretch', caption=f"Side {idx + 1}")
                except Exception:
                    with preview_cols[idx % len(preview_cols)]:
                        st.caption(f"Side {idx + 1}")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        start_scan = st.button("Scan packaging", type="primary", width='stretch', key="start_scan")
    else:
        start_scan = False
        st.info("Add at least one image, PDF, or camera capture to start an inspection.")

    if start_scan and scan_items:
        st.session_state.ui_scan_active = True
        st.session_state.ui_last_scan_id = None
        st.session_state.ui_last_report = None
        st.session_state.ui_manual_checks = {}
        st.session_state.pop("current_results", None)

        st.markdown('<div class="audit-shell">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Live mandatory declaration audit</div>', unsafe_allow_html=True)
        audit_meter = st.progress(0, text="Checking mandatory declarations…")
        audit_slot = st.empty()
        initial_checks = {rule: False for rule in RULE_LABELS}
        initial_conf = {rule: None for rule in RULE_LABELS}
        _, initial_rows = audit_progress_html(initial_checks, initial_conf, {}, finished=False)
        audit_slot.markdown(initial_rows, unsafe_allow_html=True)

        all_image_results = []
        aggregated_text_blocks = []
        all_confidences = []

        try:
            total_items = len(scan_items)
            for idx, item in enumerate(scan_items):
                extracted_text, ocr_results, confidence, image = extract_single_label_data(
                    item["bytes"],
                    conf_threshold=0.30,
                )

                if extracted_text:
                    aggregated_text_blocks.append(
                        f"--- SIDE {idx + 1} ({item['filename']}) ---\n{extracted_text}"
                    )
                    all_confidences.append(confidence)
                    all_image_results.append(
                        {
                            "filename": item["filename"],
                            "side_index": idx + 1,
                            "extracted_text": extracted_text,
                            "ocr_results": ocr_results,
                            "ocr_confidence": confidence,
                            "image": image,
                        }
                    )

                aggregated_text = "\n\n".join(aggregated_text_blocks)
                checks, field_confidence, score, status = evaluate_compliance(
                    aggregated_text,
                    all_image_results,
                )
                meter, rows = audit_progress_html(
                    checks,
                    field_confidence,
                    st.session_state.ui_manual_checks,
                    finished=False,
                )
                audit_meter.progress(
                    meter / 100,
                    text=f"Auditing side {idx + 1} of {total_items}…",
                )
                audit_slot.markdown(rows, unsafe_allow_html=True)

            if not aggregated_text_blocks:
                st.session_state.ui_scan_active = False
                st.error("No readable label text was detected. Try a clearer image or another side of the package.")
            else:
                mean_confidence = (
                    sum(all_confidences) / len(all_confidences)
                    if all_confidences else 0.0
                )
                checks, field_confidence, score, status = evaluate_compliance(
                    aggregated_text,
                    all_image_results,
                )

                # Preserve the existing backend save operation unchanged.
                batch_filename = f"Batch_{len(scan_items)}_pages_{getattr(uploaded_files[0], 'name', 'camera_capture') if uploaded_files else 'camera_capture'}"
                save_scan(
                    batch_filename,
                    score,
                    status,
                    mean_confidence,
                    aggregated_text,
                )
                scan_id_row = conn.execute("SELECT last_insert_rowid()").fetchone()
                scan_id = int(scan_id_row[0]) if scan_id_row else None
                st.session_state.ui_last_scan_id = scan_id
                if scan_id is not None:
                    initial_checks, _initial_manual_required = effective_checks(
                        checks, field_confidence, {}
                    )
                    initial_score = int(round((sum(initial_checks.values()) / len(RULE_LABELS)) * 100))
                    initial_status = "COMPLIANT" if initial_score >= 80 else "NON-COMPLIANT"
                    store_metadata(
                        scan_id,
                        st.session_state.get("current_name", "").strip() or "Unnamed Product",
                        st.session_state.get("current_category", "Others") or "Others",
                        "",
                        initial_checks,
                        initial_score,
                        initial_status,
                        {},
                    )
                st.session_state.current_results = {
                    "aggregated_text": aggregated_text,
                    "all_image_results": all_image_results,
                    "ocr_confidence": mean_confidence,
                    "checks": checks,
                    "field_confidence": field_confidence,
                    "score": score,
                    "status": status,
                }
                st.session_state.ui_scan_active = False

                final_auto_checks, manual_required = effective_checks(
                    checks,
                    field_confidence,
                    st.session_state.ui_manual_checks,
                )
                final_auto_score = int(round((sum(final_auto_checks.values()) / len(RULE_LABELS)) * 100))
                final_auto_status = "COMPLIANT" if final_auto_score >= 80 else "NON-COMPLIANT"
                meter, rows = audit_progress_html(
                    checks,
                    field_confidence,
                    st.session_state.ui_manual_checks,
                    finished=True,
                )
                audit_meter.progress(1.0, text="Audit complete")
                audit_slot.markdown(rows, unsafe_allow_html=True)
        except Exception:
            st.session_state.ui_scan_active = False
            st.error("The scan could not be completed. Please retry with the same files or clearer packaging images.")

        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # Post-scan workspace
    # -----------------------------
    data = get_current_result()
    if data:
        checks = data["checks"]
        field_confidence = data["field_confidence"]
        
        # Suppress manual requirement for MRP if it was detected
        manual_required_map = {
            rule: (False if (rule == "MRP Present" and checks.get(rule, False)) else criterion_low_conf(rule, field_confidence))
            for rule in RULE_LABELS
        }

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Review & finalize</div>', unsafe_allow_html=True)

        low_conf_rules = [r for r, required in manual_required_map.items() if required]
        if low_conf_rules:
            st.warning("Manual inspection recommended for one or more low-confidence requirements. Confirm them below only after visual inspection of the package.")

        final_check_state = {}
        for rule in RULE_LABELS:
            needs_manual = manual_required_map[rule]
            raw_pass = bool(checks.get(rule, False))
            already_manual = bool(st.session_state.ui_manual_checks.get(rule, False))
            if needs_manual:
                checked = st.checkbox(
                    f"Manual inspection completed — {rule}",
                    value=already_manual,
                    key=f"manual_{st.session_state.ui_last_scan_id}_{rule}",
                )
                st.session_state.ui_manual_checks[rule] = checked
                final_check_state[rule] = ((raw_pass and not needs_manual) or checked)
            else:
                final_check_state[rule] = raw_pass

        final_score = int(round((sum(final_check_state.values()) / len(RULE_LABELS)) * 100))
        final_status = "COMPLIANT" if final_score >= 80 else "NON-COMPLIANT"
        status_class = "pill-pass" if final_status == "COMPLIANT" else "pill-fail"

        st.text_area(
            "Inspector remark (optional)",
            value=st.session_state.get("current_remark", ""),
            placeholder="Add anything the inspector wants included in the final report…",
            key="current_remark",
            height=110,
        )

        status_left, status_right = st.columns([1.2, 2.2], gap="large")
        with status_left:
            st.markdown(
                f"<div class='mini-card'><div class='mini-label'>Final result</div><div class='mini-value'><span class='status-pill {status_class}'>{final_status}</span></div></div>",
                unsafe_allow_html=True,
            )
        with status_right:
            st.markdown(
                f"<div class='mini-card'><div class='mini-label'>Checklist</div><div class='mini-value'>{sum(final_check_state.values())} / {len(RULE_LABELS)} requirements satisfied</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        report_col, clear_col = st.columns([2.2, 1], gap="medium")
        with report_col:
            if st.button("Generate final report", type="primary", width='stretch', key="generate_report"):
                result = save_current_scan_as_final()
                if result:
                    st.success("Final report prepared and inspection saved to the database.")
        with clear_col:
            if st.button("Start over", width='stretch', key="start_over"):
                reset_inspection()
                st.rerun()

        # If a report has already been generated, retain its download button after reruns.
        if st.session_state.ui_last_report:
            current_id = st.session_state.ui_last_scan_id or "latest"
            st.download_button(
                "Download generated PDF",
                data=st.session_state.ui_last_report,
                file_name=f"compliance_report_{current_id}.pdf",
                mime="application/pdf",
                width='stretch',
                key=f"persistent_report_{current_id}",
            )

# ============================================================
# DATABASE
# ============================================================
with tab_database:
    st.markdown('<div class="section-title">Inspection database</div>', unsafe_allow_html=True)
    st.markdown('<div class="db-table-note">Every completed scan is linked to its product name, category, final compliance state, and optional inspector remark.</div>', unsafe_allow_html=True)

    try:
        history_df = load_history_dataframe()
    except Exception:
        history_df = pd.DataFrame()

    if history_df.empty:
        st.info("No completed inspections are stored yet.")
    else:
        f1, f2 = st.columns(2, gap="large")
        with f1:
            categories = ["All"] + sorted([str(x) for x in history_df["category"].dropna().unique()])
            category_filter = st.selectbox("Filter by category", categories, key="db_category_filter")
        with f2:
            status_filter = st.selectbox("Filter by compliancy", ["All", "COMPLIANT", "NON-COMPLIANT"], key="db_status_filter")

        filtered = history_df.copy()
        if category_filter != "All":
            filtered = filtered[filtered["category"] == category_filter]
        if status_filter != "All":
            filtered = filtered[filtered["status"] == status_filter]

        display_df = filtered[["id", "product_name", "category", "timestamp", "score", "status"]].copy()
        display_df.columns = ["ID", "Product", "Category", "Timestamp", "Compliance", "Status"]
        st.dataframe(display_df, width='stretch', hide_index=True)

        if not filtered.empty:
            selected_id = st.selectbox(
                "Open inspection",
                filtered["id"].tolist(),
                format_func=lambda value: f"#{value} · {filtered.loc[filtered['id'] == value, 'product_name'].iloc[0]}",
                key="db_selected_id",
            )
            selected_row = filtered[filtered["id"] == selected_id].iloc[0]

            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown(
                    f"<div class='mini-card'><div class='mini-label'>Product</div><div class='mini-value'>{html.escape(str(selected_row['product_name']))}</div><div class='audit-sub'>{html.escape(str(selected_row['category']))}</div></div>",
                    unsafe_allow_html=True,
                )
            with d2:
                selected_status = str(selected_row["status"])
                selected_class = "pill-pass" if selected_status == "COMPLIANT" else "pill-fail"
                st.markdown(
                    f"<div class='mini-card'><div class='mini-label'>Result</div><div class='mini-value'><span class='status-pill {selected_class}'>{selected_status}</span></div><div class='audit-sub'>{int(selected_row['score'])}%</div></div>",
                    unsafe_allow_html=True,
                )
            with d3:
                st.markdown(
                    f"<div class='mini-card'><div class='mini-label'>Timestamp</div><div class='mini-value' style='font-size:16px'>{html.escape(str(selected_row['timestamp']))}</div></div>",
                    unsafe_allow_html=True,
                )

            with st.expander("View inspector remark"):
                remark = str(selected_row.get("remark", "") or "").strip()
                st.write(remark if remark else "No inspector remark was recorded for this inspection.")

            with st.expander("View compliance checklist"):
                try:
                    checks_json = json.loads(selected_row.get("checks_json", "{}") or "{}")
                except Exception:
                    checks_json = {}
                for rule in RULE_LABELS:
                    if checks_json.get(rule, False):
                        st.markdown(f"✓ {rule}")
                    else:
                        st.markdown(f"× {rule}")
