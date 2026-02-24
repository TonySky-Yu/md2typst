#!/usr/bin/env python3
"""Simple GUI for md2typst converter."""

import tkinter as tk
from tkinter import font as tkfont
from md2typst import convert_document


def main():
    root = tk.Tk()
    root.title("md2typst")
    root.geometry("1200x700")

    # Top button bar
    bar = tk.Frame(root)
    bar.pack(fill=tk.X, padx=8, pady=6)

    def paste_clear():
        md_box.delete("1.0", tk.END)
        try:
            md_box.insert("1.0", root.clipboard_get())
        except tk.TclError:
            pass

    def convert():
        src = md_box.get("1.0", tk.END)
        typ_box.delete("1.0", tk.END)
        typ_box.insert("1.0", convert_document(src))

    def copy_result():
        root.clipboard_clear()
        root.clipboard_append(typ_box.get("1.0", tk.END).rstrip("\n"))

    tk.Button(bar, text="Clear & Paste", command=paste_clear).pack(side=tk.LEFT, padx=4)
    tk.Button(bar, text="Convert  \u2192", command=convert).pack(side=tk.LEFT, padx=4)
    tk.Button(bar, text="Copy Result", command=copy_result).pack(side=tk.LEFT, padx=4)

    # Two-pane text area
    panes = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashwidth=6)
    panes.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    # Use a font that supports Chinese better on Windows, fallback to generic
    # Simpler OS check
    is_windows = (root.tk.call('set', 'tcl_platform(platform)') == 'windows')
    
    # Actually, try a tuple which Tk matches better: ("Consolas", 11) with fallback handled by system often works better than strict family
    # But for Chinese specifically, specifying a CJK font is safer.
    mono = tkfont.Font(family="Microsoft YaHei" if is_windows else "Courier", size=11)

    # Left: Markdown
    left = tk.Frame(panes)
    tk.Label(left, text="Markdown", anchor=tk.W).pack(fill=tk.X)
    md_box = tk.Text(left, wrap=tk.NONE, font=mono, undo=True)
    md_box.pack(fill=tk.BOTH, expand=True)
    panes.add(left, stretch="always")

    # Right: Typst
    right = tk.Frame(panes)
    tk.Label(right, text="Typst", anchor=tk.W).pack(fill=tk.X)
    typ_box = tk.Text(right, wrap=tk.NONE, font=mono, undo=True)
    typ_box.pack(fill=tk.BOTH, expand=True)
    panes.add(right, stretch="always")

    root.mainloop()


if __name__ == "__main__":
    main()
