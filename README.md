# md2typst
ai生成的自己用的md转typst的python脚本, 没有做什么优化, 有大量边缘情况没有修正. 
主要用于减少转译工作量.

## 用法
```bash
    python md2typst.py input.md              # stdout
    python md2typst.py input.md -o out.typ   # write file
    cat file.md | python md2typst.py         # stdin
```
你也可以运行`md2typst_gui.py`打开一个简单的图形化界面来方便的进行转换. 
