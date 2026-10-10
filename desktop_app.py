"""Polished native Qt desktop interface for the crop recommendation model."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
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
    "QMainWindow { background: #f6f8f6; }"
    "QWidget { color: #1f2d22; font-family: Arial; font-size: 13px; }"
    "QFrame#header { background: #1f6b45; border-radius: 12px; }"
    "QLabel#title { color: white; font-size: 25px; font-weight: 700; }"
    "QLabel#subtitle { color: #dcf1e4; font-size: 14px; }"
    "QLabel#instruction { color: #445348; }"
    "QGroupBox { background: white; border: 1px solid #d8e1da; border-radius: 10px; margin-top: 10px; padding: 10px; font-weight: 600; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }"
    "QLabel#hint, QLabel#note, .range { color: #66756a; font-size: 12px; font-weight: 400; }"
    "QDoubleSpinBox { background: #fbfdfb; border: 1px solid #b9c9bd; border-radius: 5px; padding: 5px; min-height: 22px; }"
    "QDoubleSpinBox:focus { border: 2px solid #2e7d52; }"
    "QPushButton { background: #e8eee9; border: 1px solid #c9d5cc; border-radius: 6px; padding: 8px 13px; font-weight: 600; }"
    "QPushButton:hover { background: #dce8df; }"
    "QPushButton#primary { background: #237547; color: white; border: 1px solid #1b603a; padding: 8px 18px; }"
    "QPushButton#primary:hover { background: #195c35; }"
    "QLabel#result { font-size: 19px; font-weight: 700; color: #145a32; }"
)


class CropRecommendationWindow(QMainWindow):
    """A simple visual form for getting a crop prediction from seven inputs."""

    def __init__(self, bundle: dict) -> None:
        super().__init__()
        self.ranges = bundle["feature_ranges"]
        self.inputs: dict[str, QDoubleSpinBox] = {}
        self.setWindowTitle("Crop Recommendation")
        self.setMinimumSize(760, 650)
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
        self.inputs[feature] = spin
        return spin

    def _input_card(self, title: str, subtitle: str, features: tuple[str, ...]) -> QGroupBox:
        card = QGroupBox(title)
        card.setObjectName("inputCard")
        layout = QVBoxLayout(card)
        hint = QLabel(subtitle)
        hint.setObjectName("hint")
        layout.addWidget(hint)
        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(9)
        for feature in features:
            minimum = float(self.ranges[feature]["min"])
            maximum = float(self.ranges[feature]["max"])
            label = QLabel(f"{FEATURE_LABELS[feature]}  <span class='range'>{minimum:.1f}-{maximum:.1f}</span>")
            form.addRow(label, self._spin_box(feature))
        layout.addLayout(form)
        return card

    def _build_interface(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(15)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 17, 20, 17)
        header_title = QLabel("Crop Recommendation")
        header_title.setObjectName("title")
        header_layout.addWidget(header_title)
        header_subtitle = QLabel("A Decision Tree model for exploring soil and weather conditions")
        header_subtitle.setObjectName("subtitle")
        header_layout.addWidget(header_subtitle)
        layout.addWidget(header)

        instruction = QLabel("Fill in the measurements below. Values are limited to the ranges observed in the training data.")
        instruction.setObjectName("instruction")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        cards = QGridLayout()
        cards.setHorizontalSpacing(14)
        cards.setVerticalSpacing(14)
        cards.addWidget(
            self._input_card("Soil nutrients", "Nutrients and soil acidity", ("N", "P", "K", "ph")), 0, 0
        )
        cards.addWidget(
            self._input_card("Growing conditions", "Local climate measurements", ("temperature", "humidity", "rainfall")), 0, 1
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
        recommend = QPushButton("Get recommendation")
        recommend.setObjectName("primary")
        recommend.clicked.connect(self.recommend)
        actions.addWidget(recommend)
        layout.addLayout(actions)

        result_box = QGroupBox("Recommendation")
        result_box.setObjectName("resultCard")
        result_layout = QVBoxLayout(result_box)
        result_layout.setSpacing(9)
        self.result = QLabel("Ready for your measurements")
        self.result.setObjectName("result")
        result_layout.addWidget(self.result)
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
        """Load a valid sample used in the earlier project demonstration."""
        example = {"N": 20, "P": 75, "K": 200, "temperature": 22, "humidity": 92, "ph": 6.0, "rainfall": 110}
        for feature, value in example.items():
            self.inputs[feature].setValue(value)
        self._clear_result("Example values loaded")

    def _clear_result(self, title: str) -> None:
        self.result.setText(title)

    def recommend(self) -> None:
        """Run the saved model and display an interpretable prediction summary."""
        values = {feature: control.value() for feature, control in self.inputs.items()}
        try:
            result = predict(MODEL_PATH, values)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid measurement", str(error))
            return

        self.result.setText(f"Recommended crop: {result['prediction'].title()}")


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
