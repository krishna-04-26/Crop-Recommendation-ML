"""Polished native Qt desktop interface for the crop recommendation model."""

from __future__ import annotations

import sys
import csv
import random
from pathlib import Path

import joblib
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from crop_recommendation import FEATURE_COLUMNS
from crop_recommendation.predict import predict


BASE_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIRECTORY / "artifacts" / "decision_tree_model.joblib"
FEATURE_LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature (C)",
    "humidity": "Humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)",
}

APP_STYLE = (
    "QMainWindow, QWidget#root { background: #f5f3ec; }"
    "QWidget { color: #20352b; font-family: Arial; font-size: 13px; }"
    "QLabel { background: transparent; }"
    "QLabel#brand { font-size: 13px; font-weight: 700; letter-spacing: 3px; color: #456d53; }"
    "QLabel#title { font-size: 37px; font-weight: 700; letter-spacing: -1px; }"
    "QLabel#subtitle { color: #738078; font-size: 14px; }"
    "QLabel#badge { background: #e4eadc; color: #436147; border-radius: 12px; padding: 7px 12px; font-size: 11px; }"
    "QFrame#inputCard { background: #ffffff; border: 1px solid #e2e5db; border-radius: 18px; }"
    "QLabel#cardTitle { font-size: 19px; font-weight: 700; }"
    "QLabel#hint, QLabel#note { color: #7c877d; font-size: 11px; }"
    "QLabel#fieldLabel { font-weight: 600; font-size: 12px; }"
    "QDoubleSpinBox { background: #f7f8f3; border: 1px solid #e2e7dc; border-radius: 9px; padding: 9px 12px; min-height: 23px; font-size: 16px; }"
    "QDoubleSpinBox:focus { border: 1px solid #357451; background: #fff; }"
    "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 18px; border: none; }"
    "QPushButton { background: transparent; color: #5b6c5f; border: 1px solid #d6ddd0; border-radius: 10px; padding: 12px 18px; font-weight: 600; }"
    "QPushButton:hover { background: #e9eee3; }"
    "QPushButton#primary { background: #214e39; color: #fff; border: none; padding: 13px 25px; }"
    "QPushButton#primary:hover { background: #30684a; }"
    "QPushButton#primary:pressed { background: #163b2a; }"
    "QFrame#resultCard { background: #214e39; border-radius: 18px; }"
    "QLabel#resultCaption { color: #b5cdb5; font-size: 11px; font-weight: 700; letter-spacing: 2px; }"
    "QLabel#result { font-size: 30px; font-weight: 700; color: #faf9ec; }"
    "QLabel#resultHint { color: #c3d3c3; font-size: 12px; }"
)


class CropRecommendationWindow(QMainWindow):
    """A simple visual form for getting a crop prediction from seven inputs."""

    def __init__(self, bundle: dict) -> None:
        super().__init__()
        self.ranges = bundle["feature_ranges"]
        self.inputs: dict[str, QDoubleSpinBox] = {}
        self.examples: list[dict[str, float]] | None = None
        self.last_example_index: int | None = None
        self.setWindowTitle("Crop Recommendation")
        self.setMinimumSize(820, 790)
        self.resize(900, 850)
        self._build_interface()

    def _spin_box(self, feature: str) -> QDoubleSpinBox:
        minimum = float(self.ranges[feature]["min"])
        maximum = float(self.ranges[feature]["max"])
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(0 if feature in {"N", "P", "K"} else 2)
        spin.setSingleStep(1 if feature in {"N", "P", "K"} else 0.1)
        spin.setValue(round((minimum + maximum) / 2, spin.decimals()))
        spin.setSuffix(" " + {"temperature": "C", "humidity": "%", "rainfall": "mm"}.get(feature, ""))
        spin.setToolTip(f"Observed dataset range: {minimum:.2f} to {maximum:.2f}")
        spin.valueChanged.connect(self._measurement_changed)
        self.inputs[feature] = spin
        return spin

    def _input_card(self, title: str, subtitle: str, features: tuple[str, ...]) -> QFrame:
        card = QFrame()
        card.setObjectName("inputCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)
        hint = QLabel(subtitle)
        hint.setObjectName("hint")
        layout.addWidget(hint)
        form = QGridLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(14)
        for index, feature in enumerate(features):
            minimum = float(self.ranges[feature]["min"])
            maximum = float(self.ranges[feature]["max"])
            field = QVBoxLayout()
            field.setSpacing(5)
            label = QLabel(FEATURE_LABELS[feature])
            label.setObjectName("fieldLabel")
            field.addWidget(label)
            field.addWidget(self._spin_box(feature))
            range_label = QLabel(f"{minimum:.2f} – {maximum:.2f}")
            range_label.setObjectName("hint")
            field.addWidget(range_label)
            form.addLayout(field, index // 2, index % 2)
        layout.addLayout(form)
        return card

    def _build_interface(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(36, 30, 36, 26)
        layout.setSpacing(20)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        brand_row = QHBoxLayout()
        brand = QLabel("CROP / COMPASS")
        brand.setObjectName("brand")
        brand_row.addWidget(brand)
        brand_row.addStretch()
        badge = QLabel("SOIL → CROP")
        badge.setObjectName("badge")
        brand_row.addWidget(badge)
        header_layout.addLayout(brand_row)
        header_layout.addSpacing(12)
        header_title = QLabel("Find your next crop.")
        header_title.setObjectName("title")
        header_layout.addWidget(header_title)
        header_subtitle = QLabel("Your soil. Your climate. One clear recommendation.")
        header_subtitle.setObjectName("subtitle")
        header_layout.addWidget(header_subtitle)
        layout.addWidget(header)

        cards = QGridLayout()
        cards.setHorizontalSpacing(14)
        cards.setVerticalSpacing(14)
        cards.addWidget(
            self._input_card("01  /  Soil profile", "Nutrient levels & acidity", ("N", "P", "K", "ph")), 0, 0
        )
        cards.addWidget(
            self._input_card("02  /  Climate", "Conditions where you grow", ("temperature", "humidity", "rainfall")), 0, 1
        )
        layout.addLayout(cards)

        actions = QHBoxLayout()
        example = QPushButton("Use example values")
        example.clicked.connect(self.use_example)
        actions.addWidget(example)
        reset = QPushButton("Reset")
        reset.clicked.connect(self.reset)
        actions.addWidget(reset)
        actions.addStretch()
        recommend = QPushButton("Find my crop  →")
        recommend.setObjectName("primary")
        recommend.setCursor(Qt.PointingHandCursor)
        recommend.clicked.connect(self.recommend)
        actions.addWidget(recommend)
        layout.addLayout(actions)

        result_box = QFrame()
        result_box.setObjectName("resultCard")
        result_layout = QVBoxLayout(result_box)
        result_layout.setContentsMargins(26, 22, 26, 22)
        result_layout.setSpacing(9)
        caption = QLabel("YOUR BEST MATCH")
        caption.setObjectName("resultCaption")
        result_layout.addWidget(caption)
        self.result = QLabel("Let's find what fits.")
        self.result.setObjectName("result")
        self.result.setWordWrap(True)
        result_layout.addWidget(self.result)
        self.result_hint = QLabel("Enter your measurements, then select Find my crop.")
        self.result_hint.setObjectName("resultHint")
        result_layout.addWidget(self.result_hint)
        layout.addWidget(result_box)

        note = QLabel("Educational model only - consult local agronomic experts for real farm decisions.")
        note.setObjectName("note")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.setCentralWidget(root)

    def reset(self) -> None:
        """Restore the midpoint of each available data range."""
        for feature, control in self.inputs.items():
            minimum = float(self.ranges[feature]["min"])
            maximum = float(self.ranges[feature]["max"])
            control.setValue(round((minimum + maximum) / 2, control.decimals()))
        self._clear_result("Ready for your measurements")

    def use_example(self) -> None:
        """Choose a random real dataset sample without consecutive repeats."""
        if self.examples is None:
            try:
                with (BASE_DIRECTORY / "Crop_recommendation.csv").open(newline="", encoding="utf-8") as source:
                    self.examples = [
                        {feature: float(row[feature]) for feature in FEATURE_COLUMNS}
                        for row in csv.DictReader(source)
                    ]
                if not self.examples:
                    raise ValueError("The example dataset is empty.")
            except (OSError, ValueError, KeyError) as error:
                self.examples = None
                QMessageBox.warning(self, "Examples unavailable", str(error))
                return
        candidates = [index for index in range(len(self.examples)) if index != self.last_example_index]
        index = random.choice(candidates or [0])
        self.last_example_index = index
        example = self.examples[index]
        for feature, value in example.items():
            self.inputs[feature].setValue(value)
        self._clear_result("Random example loaded")

    def _clear_result(self, title: str) -> None:
        self.result.setText(title)
        self.result_hint.setText("Select Find my crop to get your recommendation.")

    def _measurement_changed(self) -> None:
        """Clear the previous recommendation when a measurement changes."""
        if hasattr(self, "result"):
            self._clear_result("Ready when you are.")

    def recommend(self) -> None:
        """Run the saved model and display an interpretable prediction summary."""
        values = {feature: control.value() for feature, control in self.inputs.items()}
        try:
            result = predict(MODEL_PATH, values)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid measurement", str(error))
            return

        self.result.setText(result['prediction'].title())
        self.result_hint.setText("The best predicted crop for the measurements you entered.")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python -m crop_recommendation.train --data Crop_recommendation.csv"
        )
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    window = CropRecommendationWindow(joblib.load(MODEL_PATH))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
