import tkinter as tk
import threading


def open_control_window(main_app):
    def on_forward():
        main_app.go_forward()

    def on_backward():
        main_app.go_backward()

    def on_find():
        main_app.find_and_plot_distances()

    win = tk.Tk()
    win.title("Control Panel")

    forward_frame = tk.Frame(win)
    forward_frame.pack(padx=10, pady=5, anchor='w')
    forward_btn = tk.Button(forward_frame, text="Forward", command=on_forward)
    forward_btn.pack(side='left')
    forward_label = tk.Label(forward_frame, text="(→ Arrow key)")
    forward_label.pack(side='left', padx=5)

    backward_frame = tk.Frame(win)
    backward_frame.pack(padx=10, pady=5, anchor='w')
    backward_btn = tk.Button(backward_frame, text="Backward", command=on_backward)
    backward_btn.pack(side='left')
    backward_label = tk.Label(backward_frame, text="(← Arrow key)")
    backward_label.pack(side='left', padx=5)

    find_frame = tk.Frame(win)
    find_frame.pack(padx=10, pady=5, anchor='w')
    find_btn = tk.Button(find_frame, text="Find Similar", command=on_find)
    find_btn.pack(side='left')
    find_label = tk.Label(find_frame, text="(Press 'd')")
    find_label.pack(side='left', padx=5)

    win.mainloop()


def start_control_window(main_app):
    threading.Thread(target=open_control_window, args=(main_app,), daemon=True).start()