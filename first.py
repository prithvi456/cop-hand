import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
#hello
import threading

# Initialize customtkinter appearance
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class PeopleDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("People Detection App")
        self.root.geometry("800x600")

        self.label = ctk.CTkLabel(root, text="People Detected: 0", font=("Arial", 24))
        self.label.pack(pady=10)

        self.video_label = ctk.CTkLabel(root)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.running = True
        #this is 

        # Use HOG person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.update_frame()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Resize for performance
        frame = cv2.resize(frame, (640, 480))

        # Detect people
        (regions, _) = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # Draw rectangles around detected people
        for (x, y, w, h) in regions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update the label with number of people
        self.label.configure(text=f"People Detected: {len(regions)}")

        # Convert BGR to RGB and then to ImageTk format
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def on_close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    app = PeopleDetectorApp(root)
    root.mainloop()
