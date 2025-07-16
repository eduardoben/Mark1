import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QLabel, QFileDialog, QHBoxLayout, QLineEdit, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
from PyQt5.QtGui import QFont
import gc

import requests

import classifier 


def image_classification(image_path):
    return classifier.inference(image_path)


def query_llm(classification, user_message):
    prompt = f"""You are a medical assistant expert in dermatology. The user has uploaded an image that was classified as: {classification}.
    Based on that, provide a clear and understandable medical explanation (for a general patient) about this skin lesion: {classification}.
    Include information about symptoms, risks, when to see a dermatologist and possible treatments if necessary.
    """

    payload = {
        "model": "gpt-4o",  # Or the model you have loaded in LocalAI
        "messages": [
            {"role": "system", "content": "You are a dermatology expert who explains medical information clearly and briefly for patients."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = None
    try:
        response = requests.post("http://localhost:1230/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        return f"Error querying local LLM: {e}"
    except (KeyError, IndexError):
        return "Error interpreting model response."
    finally:
        # Clean up response object
        if response is not None:
            response.close()
            del response
        gc.collect()


class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classification Chat")
        self.resize(700, 600)

        # Common font
        font_size = QFont("Arial", 12)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Image
        self.image_label = QLabel("Loaded image:")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setFixedSize(300, 300)
        self.image_label.setFont(font_size)  
        self.layout.addWidget(self.image_label)

        # Button layout for Load Image and Clear Chat
        self.button_layout = QHBoxLayout()
        
        # Load image button
        self.upload_button = QPushButton("Load Image")
        self.upload_button.setFont(font_size)
        self.upload_button.setStyleSheet("background-color: lightblue;")
        self.upload_button.setFixedHeight(40)
        self.upload_button.clicked.connect(self.load_image)
        self.button_layout.addWidget(self.upload_button)

        # Clear chat button
        self.clear_chat_button = QPushButton("Clear Chat")
        self.clear_chat_button.setFont(font_size)
        self.clear_chat_button.setStyleSheet("background-color: lightcoral;")
        self.clear_chat_button.setFixedHeight(40)
        self.clear_chat_button.clicked.connect(self.cleanup_previous_text)
        self.button_layout.addWidget(self.clear_chat_button)
        
        self.layout.addLayout(self.button_layout)

        # Chat
        self.chat_box = QTextEdit()
        self.chat_box.setReadOnly(True)
        self.chat_box.setStyleSheet("border: 1px solid black;")
        self.chat_box.setFixedHeight(200)
        self.chat_box.setAlignment(Qt.AlignTop)
        self.chat_box.setFont(font_size)
        self.layout.addWidget(self.chat_box)

        # Text input and send button
        self.input_layout = QHBoxLayout()
        self.input_layout.setContentsMargins(0, 0, 0, 0)
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        self.user_input.setStyleSheet("border: 1px solid black;")
        self.user_input.setFont(font_size)

        self.send_button = QPushButton("Send")
        self.send_button.setFont(font_size)
        self.send_button.setStyleSheet("background-color: lightgreen;")
        self.send_button.setFixedHeight(40)
        self.send_button.setFixedWidth(100)
        self.send_button.setToolTip("Send your message to the assistant")
        self.send_button.setCursor(Qt.PointingHandCursor)        
        self.send_button.clicked.connect(self.send_message)

        self.input_layout.addWidget(self.user_input)
        self.input_layout.setAlignment(Qt.AlignLeft)
        self.input_layout.addWidget(self.send_button)
        self.input_layout.setContentsMargins(10, 10, 10, 10)
        self.input_layout.setAlignment(Qt.AlignRight)  
        self.input_layout.setSpacing(10)
        self.layout.addLayout(self.input_layout)

        # State
        self.classification = None
        # Path to the image
        self.image_path = None
        # Store current pixmap for cleanup
        self.current_pixmap = None

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select an image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            # Clean up previous image data
            self.cleanup_previous_image()
            
            self.image_path = path
            pixmap = QPixmap(path).scaled(300, 300, Qt.KeepAspectRatio)
            
            # Check if the image is valid
            if pixmap.isNull():
                QMessageBox.critical(self, "Image Error", "The selected image is invalid or corrupted.")
                return
            
            # Store current pixmap reference for cleanup
            self.current_pixmap = pixmap
            
            # Set the image in the label
            self.image_label.setText("")  # Clear previous text
            self.image_label.setPixmap(pixmap)

            try:
                # Classify image
                self.classification = image_classification(path)
                if not self.classification:
                    raise ValueError("Classification returned an empty result.")
                
                # Write classification result to chat
                self.write_chat(f"[Classifier] Image classified as: {self.classification}")

                # Send automatic question to LLM
                automatic_question = f"Provide me with clear, brief but detailed medical information about the following skin lesion: {self.classification}."
                response = query_llm(self.classification, automatic_question)
                if not response:
                    response = "Could not get a response from the model."
                self.write_chat(f"[Assistant]: {response}")
                
            except Exception as e:
                QMessageBox.critical(self, "Classification Error", f"Could not classify image:\n{e}")
            finally:
                # Force garbage collection after classification
                gc.collect()

    def send_message(self):
        """Handle sending a message to the LLM."""
        text = self.user_input.text().strip()
        # Check if the input is empty
        if not text:
            return

        if not self.classification:
            QMessageBox.warning(self, "Missing Image", "You must load an image first.")
            self.write_chat("[Error]: You must load an image first.")
            return

        self.write_chat(f"[User]: {text}")
        self.user_input.clear()
        
        try:
            response = query_llm(self.classification, text)
            if not response:
                response = "Could not get a response from the model."

            self.write_chat(f"[Assistant]: {response}")

        except Exception as e:
            self.write_chat(f"[Error querying LLM]: {e}")
        finally:
            # Clean up after each message
            gc.collect()

    def write_chat(self, text):
        """Append text to the chat box."""
        self.chat_box.append(text)
        self.chat_box.verticalScrollBar().setValue(self.chat_box.verticalScrollBar().maximum())

    def cleanup_previous_image(self):
        """Clean up previous image data from memory."""
        if self.current_pixmap is not None:
            del self.current_pixmap
            self.current_pixmap = None
        
        # Clear the image label
        self.image_label.clear()
        self.image_label.setText("Loaded image:")
        
        # Reset classification
        self.classification = None
        self.image_path = None

        
        # Force garbage collection
        if hasattr(self, 'image'):
            self.image.close()
            del self.image
        gc.collect()

    def cleanup_previous_text(self):
        """Clear the chat box and reset conversation state."""
        # Ask for confirmation before clearing
        reply = QMessageBox.question(
            self, 
            'Clear Chat', 
            'Are you sure you want to clear the chat history?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clear the chat box
            self.chat_box.clear()
            
            # Clear any text in the input field
            self.user_input.clear()
            
            # Optionally reset the classification state
            # (Comment out these lines if you want to keep the image and classification)
            # self.classification = None
            # self.image_path = None
            
            # Show confirmation message
            self.write_chat("[System]: Chat history cleared.")
            
            # Force garbage collection to free up memory
            gc.collect()

    def closeEvent(self, event):
        """Handle application close event - clean up all resources."""
        try:
            # Clean up current image
            self.cleanup_previous_image()
            
            # Clean up the model from memory
            if hasattr(classifier, 'cleanup_model'):
                classifier.cleanup_model()
                self.write_chat("[System]: Model cleaned up from memory")
            
            # Clear chat history
            self.chat_box.clear()
            
            # Force final garbage collection
            gc.collect()
            
            # Accept the close event
            event.accept()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            event.accept()  # Still close the application

    def __del__(self):
        """Destructor to ensure cleanup when object is destroyed."""
        try:
            self.cleanup_previous_image()
            if hasattr(classifier, 'cleanup_model'):
                classifier.cleanup_model()
                self.write_chat("[System]: Model cleaned up from memory")
        except:
            pass  # Ignore errors during destruction

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatApp()
    window.setStyleSheet("background-color: #f0f0f0;")
    window.setFont(QFont("Arial", 12))
    window.setWindowTitle("Image Classification Chat")
    window.resize(700, 600)
    window.show()
    
    try:
        # Start the application event loop
        sys.exit(app.exec_())

    finally:
        # Final cleanup when application exits
        if hasattr(classifier, 'cleanup_model'):
            classifier.cleanup_model()
        gc.collect()