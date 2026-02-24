#!/usr/bin/env python3
r"""
md2typst.py  -  Convert Markdown + LaTeX math to Typst
AST-based Object-Oriented Edition

Usage:
    python md2typst.py input.md              # stdout
    python md2typst.py input.md -o out.typ   # write file
    cat file.md | python md2typst.py         # stdin
"""

import re, sys, argparse

ACCENTS = {
    r'\bar': 'overline', r'\hat': 'hat', r'\tilde': 'tilde',
    r'\dot': 'dot', r'\ddot': 'dot.double', r'\dddot': 'dot.triple',
    r'\ddddot': 'dot.quad', r'\vec': 'arrow', r'\overrightarrow': 'arrow',
    r'\overleftarrow': 'arrow.l', r'\widehat': 'hat', r'\widetilde': 'tilde',
    r'\overline': 'overline', r'\underline': 'underline',
}
FONTS = {
    r'\mathbb': 'bb', r'\mathbf': 'bold', r'\mathcal': 'cal',
    r'\mathrm': 'upright', r'\mathsf': 'sans', r'\mathtt': 'mono',
    r'\mathit': 'italic', r'\mathfrak': 'frak', r'\mathscr': 'scr',
    r'\boldsymbol': 'bold', r'\textbf': 'bold',
}
GREEK = {
    r'\varepsilon':'epsilon.alt',r'\vartheta':'theta.alt',r'\varpi':'pi.alt',
    r'\varrho':'rho.alt',r'\varsigma':'sigma.alt',r'\varphi':'phi.alt',
    r'\varkappa':'kappa.alt',
    r'\phi':'phi',
    r'\alpha':'alpha',r'\beta':'beta',r'\gamma':'gamma',r'\delta':'delta',
    r'\epsilon':'epsilon',r'\zeta':'zeta',r'\eta':'eta',r'\theta':'theta',
    r'\iota':'iota',r'\kappa':'kappa',r'\lambda':'lambda',r'\mu':'mu',
    r'\nu':'nu',r'\xi':'xi',r'\pi':'pi',r'\rho':'rho',r'\sigma':'sigma',
    r'\tau':'tau',r'\upsilon':'upsilon',r'\chi':'chi',r'\psi':'psi',r'\omega':'omega',
    r'\Gamma':'Gamma',r'\Delta':'Delta',r'\Theta':'Theta',r'\Lambda':'Lambda',
    r'\Xi':'Xi',r'\Pi':'Pi',r'\Sigma':'Sigma',r'\Upsilon':'Upsilon',
    r'\Phi':'Phi',r'\Psi':'Psi',r'\Omega':'Omega',
}
TOKENS_MAP = {
    # 箭头
    r'\rightarrow': 'arrow.r', r'\to': 'arrow.r',
    r'\leftarrow': 'arrow.l', r'\gets': 'arrow.l',
    r'\Rightarrow': 'arrow.r.double', r'\Leftarrow': 'arrow.l.double',
    r'\leftrightarrow': 'arrow.l.r', r'\Leftrightarrow': 'arrow.l.r.double',
    r'\longrightarrow': 'arrow.r.long', r'\longleftarrow': 'arrow.l.long',
    r'\Longleftrightarrow': 'arrow.l.r.double.long',
    r'\uparrow': 'arrow.t', r'\downarrow': 'arrow.b',
    r'\Uparrow': 'arrow.t.double', r'\Downarrow': 'arrow.b.double',
    r'\updownarrow': 'arrow.t.b', r'\Updownarrow': 'arrow.t.b.double',
    r'\nearrow': 'arrow.tr', r'\searrow': 'arrow.br',
    r'\swarrow': 'arrow.bl', r'\nwarrow': 'arrow.tl',
    r'\mapsto': 'arrow.r.bar', r'\longmapsto': 'arrow.r.long.bar',
    r'\hookrightarrow': 'arrow.r.hook', r'\hookleftarrow': 'arrow.l.hook',
    r'\rightharpoonup': 'harpoon.rt', r'\rightharpoondown': 'harpoon.rb',
    r'\leftharpoonup': 'harpoon.lt', r'\leftharpoondown': 'harpoon.lb',
    r'\rightleftharpoons': 'harpoons.rtlb', r'\leftrightharpoons': 'harpoons.ltrb',
    r'\iff': 'arrow.l.r.double.long', r'\implies': '=>', r'\impliedby': '<=',
    # 关系运算符
    r'\neq': 'eq.not', r'\ne': 'eq.not',
    r'\equiv': 'eq.triple', r'\approx': 'approx',
    r'\cong': 'tilde.equiv', r'\simeq': 'tilde.eq', r'\sim': 'tilde.op',
    r'\propto': 'prop',
    r'\in': 'in', r'\notin': 'in.not', r'\ni': 'in.rev',
    r'\subset': 'subset', r'\supset': 'supset',
    r'\subseteq': 'subset.eq', r'\supseteq': 'supset.eq',
    r'\subsetneq': 'subset.neq', r'\supsetneq': 'supset.neq',
    r'\cap': 'inter', r'\cup': 'union',
    r'\bigcap': 'inter.big', r'\bigcup': 'union.big',
    r'\emptyset': 'emptyset', r'\varnothing': 'emptyset',
    r'\forall': 'forall', r'\exists': 'exists', r'\nexists': 'exists.not',
    r'\neg': 'not', r'\lnot': 'not',
    # 逻辑运算符
    r'\land': 'and', r'\wedge': 'and',
    r'\lor': 'or', r'\vee': 'or',
    r'\bigwedge': 'and.big', r'\bigvee': 'or.big',
    r'\oplus': 'xor', r'\bigoplus': 'xor.big',
    # 集合论
    r'\setminus': 'without', r'\smallsetminus': 'without',
    r'\times': 'times',
    r'\otimes': 'times.o', r'\bigotimes': 'times.o.big',
    r'\sqcap': 'inter.sq', r'\sqcup': 'union.sq',
    r'\bigsqcap': 'inter.sq.big', r'\bigsqcup': 'union.sq.big',
    # 微积分
    r'\int': 'integral', r'\iint': 'integral.double',
    r'\iiint': 'integral.triple', r'\oint': 'integral.cont',
    r'\partial': 'partial', r'\nabla': 'nabla',
    r'\sum': 'sum', r'\prod': 'product', r'\coprod': 'product.co',
    # 括号
    r'\langle': 'chevron.l', r'\rangle': 'chevron.r',
    r'\lfloor': 'floor.l', r'\rfloor': 'floor.r',
    r'\lceil': 'ceil.l', r'\rceil': 'ceil.r',
    r'\lvert': 'bar.v', r'\rvert': 'bar.v',
    r'\lVert': 'bar.v.double', r'\rVert': 'bar.v.double',
    # 二元运算符
    r'\pm': 'plus.minus', r'\mp': 'minus.plus', r'\div': 'div',
    r'\cdot': 'dot.op', r'\ast': 'ast.op',
    r'\circ': 'circle.stroked.tiny', r'\bullet': 'bullet', r'\star': 'star.op',
    r'\dagger': 'dagger', r'\ddagger': 'dagger.double',
    r'\triangleleft': 'triangle.stroked.l', r'\triangleright': 'triangle.stroked.r',
    r'\triangle': 'triangle.stroked.t',
    r'\bigtriangleup': 'triangle.stroked.t', r'\bigtriangledown': 'triangle.stroked.b',
    r'\square': 'square.stroked', r'\Box': 'square.stroked',
    r'\blacksquare': 'square.filled',
    r'\blacktriangle': 'triangle.filled.small.t',
    r'\blacktriangledown': 'triangle.filled.small.b',
    r'\blacktriangleleft': 'triangle.filled.small.l',
    r'\blacktriangleright': 'triangle.filled.small.r',
    # 比较运算符
    r'\leq': 'lt.eq', r'\le': 'lt.eq', r'\geq': 'gt.eq', r'\ge': 'gt.eq',
    r'\ll': 'lt.double', r'\gg': 'gt.double',
    r'\lll': 'lt.triple', r'\ggg': 'gt.triple',
    r'\prec': 'prec', r'\succ': 'succ',
    r'\preceq': 'prec.eq', r'\succeq': 'succ.eq',
    r'\parallel': 'parallel', r'\nparallel': 'parallel.not',
    r'\perp': 'perp', r'\mid': 'divides', r'\nmid': 'divides.not',
    # 标点符号
    r'\dots': 'dots.h', r'\ldots': 'dots.h',
    r'\cdots': 'dots.h.c', r'\vdots': 'dots.v', r'\ddots': 'dots.down',
    # 特殊符号
    r'\infty': 'oo',
    r'\aleph': 'aleph', r'\beth': 'beth', r'\gimel': 'gimel', r'\daleth': 'daleth',
    r'\hbar': 'planck',
    r'\imath': 'dotless.i', r'\jmath': 'dotless.j',
    r'\ell': 'ell', r'\wp': 'nothing',
    r'\Re': 'Re', r'\Im': 'Im',
    r'\angle': 'angle', r'\measuredangle': 'angle.arc', r'\sphericalangle': 'angle.spheric',
    r'\degree': 'degree',
    r'\prime': 'prime',
    r'\top': 'tack.b', r'\bot': 'tack.t',
    r'\models': 'models',
    r'\vdash': 'tack.r', r'\dashv': 'tack.l',
    r'\Vdash': 'tack.r.double', r'\Vvdash': 'forces',
    r'\nvdash': 'tack.r.not', r'\nvDash': 'tack.r.double.not',
    r'\therefore': 'therefore', r'\because': 'because',
    r'\QED': 'qed',
    r'\smile': 'smile', r'\frown': 'frown',
    r'\copyright': 'copyright',
    r'\checkmark': 'checkmark', r'\maltese': 'maltese', r'\diameter': 'diameter',
    r'\|': 'bar.v.double',
    r'\&': 'amp', r'\%': 'percent', r'\#': 'hash', r'\$': 'dollar',
    r'\euro': 'euro', r'\pounds': 'pound', r'\sterling': 'pound', r'\yen': 'yen',
    # 音乐符号
    r'\flat': 'flat', r'\natural': 'natural', r'\sharp': 'sharp',
    # 扑克牌花色
    r'\clubsuit': 'suit.club.filled', r'\diamondsuit': 'suit.diamond.stroked',
    r'\heartsuit': 'suit.heart.stroked', r'\spadesuit': 'suit.spade.filled',
    # 空格
    r'\,': 'space.thin', r'\:': 'space.med', r'\;': 'space.med',
    r'\quad': 'quad', r'\qquad': 'space.quad',
    r'\ ': 'space.nobreak',
    r'\!': '',
    # 三角函数 / 数学函数
    r'\sin': 'sin', r'\cos': 'cos', r'\tan': 'tan', r'\cot': 'cot',
    r'\sec': 'sec', r'\csc': 'csc',
    r'\arcsin': 'arcsin', r'\arccos': 'arccos', r'\arctan': 'arctan',
    r'\sinh': 'sinh', r'\cosh': 'cosh', r'\tanh': 'tanh', r'\coth': 'coth',
    r'\exp': 'exp', r'\log': 'log', r'\ln': 'ln',
    r'\det': 'det', r'\dim': 'dim', r'\ker': 'ker', r'\hom': 'hom',
    r'\arg': 'arg', r'\deg': 'deg', r'\gcd': 'gcd',
    r'\max': 'max', r'\min': 'min', r'\sup': 'sup', r'\inf': 'inf',
    r'\Pr': 'Pr', r'\lim': 'lim', r'\liminf': 'liminf', r'\limsup': 'limsup',
    r'\\': '\\'
}

# 非命令符号映射（SYM 类型 token）
SYMS_MAP = {
    '~': 'space.nobreak',
}

# -- AST Nodes --

class Expr:
    def __init__(self, children): self.children = children
class Literal:
    def __init__(self, text): self.text = text
class Cmd:
    def __init__(self, name, args): self.name = name; self.args = args
class Group:
    def __init__(self, children): self.children = children
class SubSup:
    def __init__(self, base, sub=None, sup=None): self.base = base; self.sub = sub; self.sup = sup
class Env:
    def __init__(self, name, content): self.name = name; self.content = content

# -- Lexical Analysis --

def tokenize(s):
    tokens = []
    pattern = re.compile(r'(\\begin\{[^\}]+\})|(\\end\{[^\}]+\})|(%[^\n]*)|(\\[a-zA-Z]+)|(\\[^a-zA-Z])|([0-9]+(?:\.[0-9]+)?)|([a-zA-Z])|([^a-zA-Z0-9\\\s%])')
    pos = 0
    while pos < len(s):
        m = pattern.search(s, pos)
        if not m: break
        if m.start() > pos: pass
        if m.group(1): tokens.append(('BEGIN', m.group(1)[7:-1]))
        elif m.group(2): tokens.append(('END', m.group(2)[5:-1]))
        elif m.group(3): tokens.append(('COMMENT', m.group(3)))
        elif m.group(4): tokens.append(('CMD', m.group(4)))
        elif m.group(5): tokens.append(('ESC', m.group(5)))
        elif m.group(6): tokens.append(('NUM', m.group(6)))
        elif m.group(7): tokens.append(('LET', m.group(7)))
        elif m.group(8): tokens.append(('SYM', m.group(8)))
        pos = m.end()
    return tokens

# -- AST to Typst String --

def _compact(node, env=None):
    """Render a node with children joined WITHOUT spaces.
    Used for \\text{...} and \\operatorname{...} whose content is
    split into individual letter tokens by the tokenizer."""
    if type(node) in (Group, Expr):
        return ''.join(filter(None, (_compact(c, env) for c in node.children)))
    return to_typst(node, env)

def to_typst(node, current_env=None):
    if type(node) is Expr:
        parts = [to_typst(c, current_env) for c in node.children]
        s = ' '.join(p for p in parts if p)
        s = s.replace(' ,', ',').replace(' .', '.').replace(' !', '!')
        
        if current_env in ('pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix', 'matrix', 'array'):
            s = s.replace(' & ', ', ').replace('&', ',')
            s = s.replace(' \\ ', '; ')
        elif current_env == 'cases':
            s = s.replace(' \\ ', ', ')
            s = s.replace(',  ', ', ')
        elif current_env in ('aligned', 'align', 'gather', 'split', 'aligned*', 'align*', 'equation', 'equation*'):
            s = s.replace(' \\ ', ' \\\n')
        return s
    elif type(node) is Literal:
        if node.text.startswith('%'): return '// ' + node.text[1:]
        if node.text == r'\.': return ''
        if node.text in SYMS_MAP: return SYMS_MAP[node.text]
        return node.text
    elif type(node) is tuple:
        typ, val = node
        if typ == 'CMD':
            if val in TOKENS_MAP: return TOKENS_MAP[val]
            if val in GREEK: return GREEK[val]
        return val
    elif type(node) is Group:
        return ' '.join(filter(None, [to_typst(c, current_env) for c in node.children]))
    elif type(node) is SubSup:
        res = to_typst(node.base, current_env)
        if node.sub:
            sub_s = to_typst(node.sub, current_env)
            if type(node.sub) is Literal or (type(node.sub) is Group and len(node.sub.children) == 1 and type(node.sub.children[0]) is Literal):
                res += f"_{sub_s}"
            else: res += f"_({sub_s})"
        if node.sup:
            sup_s = to_typst(node.sup, current_env)
            if type(node.sup) is Literal or (type(node.sup) is Group and len(node.sup.children) == 1 and type(node.sup.children[0]) is Literal):
                res += f"^{sup_s}"
            else: res += f"^({sup_s})"
        return res
    elif type(node) is Cmd:
        if node.name in (r'\frac', r'\dfrac', r'\tfrac', r'\cfrac'):
            return f"frac({to_typst(node.args[0], current_env)}, {to_typst(node.args[1], current_env)})"
        elif node.name in (r'\overset', r'\underset', r'\stackrel'):
            func = 't' if node.name != r'\underset' else 'b'
            return f"attach({to_typst(node.args[1], current_env)}, {func}: {to_typst(node.args[0], current_env)})"
        elif node.name == r'\sqrt':
            if len(node.args) == 2: return f"root({to_typst(node.args[1], current_env)}, {to_typst(node.args[0], current_env)})"
            return f"sqrt({to_typst(node.args[0], current_env)})"
        elif node.name in ACCENTS: return f"{ACCENTS[node.name]}({to_typst(node.args[0], current_env)})"
        elif node.name in FONTS: return f"{FONTS[node.name]}({to_typst(node.args[0], current_env)})"
        elif node.name == r'\boxed': return f"rect({to_typst(node.args[0], current_env)})"
        elif node.name == r'\operatorname': return f'op("{_compact(node.args[0], current_env)}")'
        elif node.name == r'\text': return f'"{_compact(node.args[0], current_env)}"'
        elif node.name == r'\abs': return f"abs({to_typst(node.args[0], current_env)})"
        elif node.name == r'\norm': return f"norm({to_typst(node.args[0], current_env)})"
        elif node.name == r'\lr':
            l_val = to_typst(node.args[0], current_env) if node.args[0] else ''
            r_val = to_typst(node.args[2], current_env) if node.args[2] else ''
            if l_val == r'\.': l_val = ''
            if r_val == r'\.': r_val = ''
            return f"lr({l_val} {to_typst(node.args[1], current_env)} {r_val})"
        else:
            if node.name in (r'\big', r'\Big', r'\bigg', r'\Bigg', r'\bigl', r'\Bigl', r'\biggl', r'\Biggl', r'\bigr', r'\Bigr', r'\biggr', r'\Biggr', r'\limits', r'\nolimits', r'\displaystyle', r'\textstyle', r'\scriptstyle', r'\scriptscriptstyle'):
                return ''
            if node.name in GREEK: return GREEK[node.name]
            if node.name in TOKENS_MAP: return TOKENS_MAP[node.name]
            return node.name[1:] if node.name.startswith('\\') else node.name
    elif type(node) is Env:
        inner = to_typst(node.content, node.name)
        if node.name in ('pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix', 'matrix', 'array'):
            d = {'pmatrix':'"(", ', 'bmatrix':'"[", ', 'vmatrix':'"|", ', 'Vmatrix':'"||", ', 'matrix':'none, ', 'array':'none, '}[node.name]
            return f"mat(delim: {d}{inner})"
        elif node.name == 'cases': return f"cases({inner})"
        elif node.name in ('aligned', 'align', 'gather', 'split', 'equation', 'aligned*', 'align*', 'equation*'):
            return inner
        else: return f"{node.name}({inner})"

# -- Parsing Logic --

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.tags = []

    def peek(self): return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    def consume(self):
        tok = self.peek()
        if tok: self.pos += 1
        return tok

    def parse_arg(self):
        tok = self.peek()
        if not tok: return None
        if tok[0] == 'SYM' and tok[1] == '{':
            self.consume()
            expr = self.parse_expr(end_tokens=['}'])
            if self.peek() and self.peek()[0] == 'SYM' and self.peek()[1] == '}': self.consume()
            return Group(expr.children)
        return self.parse_atom()

    def parse_atom(self):
        tok = self.consume()
        if not tok: return None
        typ, val = tok
        if typ == 'BEGIN':
            if val == 'array':
                next_tok = self.peek()
                if next_tok and next_tok[0] == 'SYM' and next_tok[1] == '{':
                    self.consume()
                    self.parse_expr(end_tokens=['}'])
                    if self.peek() and self.peek()[0] == 'SYM' and self.peek()[1] == '}': self.consume()
            content = self.parse_expr(end_env=val)
            if self.peek() and self.peek()[0] == 'END' and self.peek()[1] == val: self.consume()
            return Env(val, content)
        if typ == 'SYM' and val == '{':
            content = self.parse_expr(end_tokens=['}'])
            if self.peek() and self.peek()[0] == 'SYM' and self.peek()[1] == '}': self.consume()
            return Group(content.children)
        if typ == 'CMD' or typ == 'ESC':
            if val in [r'\tag', r'\label']:
                arg = self.parse_arg()
                self.tags.append(to_typst(arg))
                return Literal('')
            if val in [r'\frac', r'\dfrac', r'\tfrac', r'\cfrac', r'\overset', r'\underset', r'\stackrel']:
                return Cmd(val, [self.parse_arg(), self.parse_arg()])
            elif val == r'\sqrt':
                next_tok = self.peek()
                if next_tok and next_tok[0] == 'SYM' and next_tok[1] == '[':
                    self.consume()
                    deg = self.parse_expr(end_tokens=[']'])
                    if self.peek() and self.peek()[0] == 'SYM' and self.peek()[1] == ']': self.consume()
                    return Cmd(val, [self.parse_arg(), deg])
                return Cmd(val, [self.parse_arg()])
            elif val in ACCENTS or val in FONTS or val in [r'\boxed', r'\text', r'\operatorname']:
                return Cmd(val, [self.parse_arg()])
            else: return Cmd(val, [])
        return Literal(val)

    def parse_expr(self, end_tokens=None, end_env=None):
        end_tokens = end_tokens or []
        children = []
        while self.peek():
            tok = self.peek()
            typ, val = tok
            if end_env and typ == 'END' and val == end_env: break
            if typ == 'SYM' and val in end_tokens: break
            if typ == 'CMD' and val == r'\right': break
            if typ == 'CMD' and val == r'\left':
                self.consume()
                delimL = self.consume()
                content = self.parse_expr()
                delimR = None
                if self.peek() and self.peek()[0] == 'CMD' and self.peek()[1] == r'\right':
                    self.consume()
                    delimR = self.consume()
                l_val = delimL[1] if delimL else ''
                r_val = delimR[1] if delimR else ''
                if l_val in ('|', r'\lvert', r'\mid') and r_val in ('|', r'\rvert', r'\mid'):
                    children.append(Cmd(r'\abs', [content]))
                elif l_val in (r'\|', r'\lVert') and r_val in (r'\|', r'\rVert'):
                    children.append(Cmd(r'\norm', [content]))
                else: children.append(Cmd(r'\lr', [delimL, content, delimR]))
                continue
            if typ == 'SYM' and val in ['_', '^']:
                self.consume()
                arg = self.parse_arg()
                if children:
                    prev = children[-1]
                    if isinstance(prev, SubSup):
                        if val == '_' and not prev.sub: prev.sub = arg
                        elif val == '^' and not prev.sup: prev.sup = arg
                        else: children[-1] = SubSup(prev, sub=arg if val == '_' else None, sup=arg if val == '^' else None)
                    else: children[-1] = SubSup(prev, sub=arg if val == '_' else None, sup=arg if val == '^' else None)
                else: children.append(SubSup(Literal(''), sub=arg if val == '_' else None, sup=arg if val == '^' else None))
                continue
            children.append(self.parse_atom())
        return Expr([c for c in children if type(c) is not Literal or c.text != ''])

# -- Integration & Document Wrapping --

def convert_math(tex, is_display=False):
    p = Parser(tokenize(tex))
    ast = p.parse_expr()
    
    is_boxed = False
    if len(ast.children) == 1 and type(ast.children[0]) is Cmd and ast.children[0].name == r'\boxed':
        is_boxed = True
        ast = ast.children[0].args[0]
        
    typst_math = to_typst(ast)
    tags = p.tags
    
    if is_boxed and is_display:
        return f"#rect($ \n{typst_math}\n $) " + " ".join(f"<{t}>" for t in tags)
    elif is_boxed:
        return f"#rect[$ {typst_math} $]"
    else:
        res = f"$\n{typst_math}\n$" if is_display else f"${typst_math}$"
        if is_display and tags: res += " " + " ".join(f"<{t}>" for t in tags)
        return res

def _conv_heading(line):
    m = re.match(r'^(#{1,6})\s+(.*)', line)
    if not m:
        return line
    heading_text = convert_text_line(m.group(2).strip())
    return '=' * len(m.group(1)) + ' ' + heading_text

def _conv_markup(text):
    # Bold: **…** / __…__
    # Opening delimiter must NOT be followed by whitespace;
    # closing delimiter must NOT be preceded by whitespace.
    # This prevents "a ** b ** c" from being treated as bold.
    text = re.sub(r'\*\*(?!\s)(.+?)(?<!\s)\*\*', '\x01\\1\x01', text)
    text = re.sub(r'__(?!\s)(.+?)(?<!\s)__',     '\x01\\1\x01', text)
    # Italic: *…*
    # Opening * : not preceded by *, not followed by whitespace or *
    # Closing * : not preceded by whitespace or *, not followed by *
    # This prevents "a * b * c" (spaces around *) from becoming italic.
    text = re.sub(r'(?<!\*)\*(?![\s*])(.+?)(?<![\s*])\*(?!\*)', r'_\1_', text)
    # Escape remaining literal * for Typst (lone stars, unmatched stars, etc.)
    text = text.replace('*', '\\*')
    # Restore Typst bold markers (placeholders → *)
    text = text.replace('\x01', '*')
    return text

def _conv_list(line):
    m = re.match(r'^(\s*)([-*+])\s+(.*)', line)
    if m: return '  ' * (len(m.group(1)) // 2) + '- ' + m.group(3)
    m = re.match(r'^(\s*)\d+\.\s+(.*)', line)
    if m: return '  ' * (len(m.group(1)) // 2) + '+ ' + m.group(2)
    return line

def _conv_links(line):
    line = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'#image("\2", alt: "\1")', line)
    line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',  r'#link("\2")[\1]', line)
    return line

def convert_text_line(line):
    line = _conv_links(line)
    # Split out inline code spans first – pass them through verbatim
    segments = re.split(r'(`[^`]+`)', line)
    result = []
    for seg in segments:
        if seg.startswith('`') and seg.endswith('`'):
            result.append(seg)
            continue
        # Also capture \[...\] (display) and \(...\) (inline) delimiters.
        # The $ patterns are unchanged; the new alternatives are appended.
        _MATH_RE = re.compile(
            r'(?<!\\)('
            r'\$\$(?:[^$\\]|\\.)+?\$\$'
            r'|\$(?:[^$\\]|\\.)+?\$'
            r'|\\\[[\s\S]+?\\\]'
            r'|\\\([\s\S]+?\\\)'
            r')'
        )
        parts = _MATH_RE.split(seg)
        for part in parts:
            if part.startswith('$$') and part.endswith('$$') and len(part) >= 5:
                result.append(convert_math(part[2:-2], is_display=True))
            elif part.startswith('$') and part.endswith('$') and len(part) >= 3:
                result.append(convert_math(part[1:-1], is_display=False))
            elif part.startswith(r'\[') and part.endswith(r'\]'):
                result.append(convert_math(part[2:-2], is_display=True))
            elif part.startswith(r'\(') and part.endswith(r'\)'):
                result.append(convert_math(part[2:-2], is_display=False))
            else:
                p = _conv_list(part)
                p = _conv_markup(p)
                result.append(p)
    return ''.join(result)

def convert_document(text):
    lines = text.split('\n')
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if line.startswith('#'):
            out.append(_conv_heading(line)); i += 1; continue
        if stripped.startswith('```'):
            lang = stripped[3:].strip()
            code_lines = []; i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i]); i += 1
            code = '\n'.join(code_lines)
            out.append(f'```{lang}\n{code}\n```' if lang else f'```\n{code}\n```')
            i += 1; continue
        if stripped == '$$':
            math_lines = []; i += 1
            while i < len(lines) and lines[i].strip() != '$$':
                math_lines.append(lines[i]); i += 1
            out.append(convert_math('\n'.join(math_lines), is_display=True)); i += 1; continue
        m = re.match(r'^\s*\$\$(.*)\$\$\s*$', line, re.DOTALL)
        if m and m.group(1):
            out.append(convert_math(m.group(1), is_display=True)); i += 1; continue
        if stripped.startswith('$$') and not stripped.endswith('$$'):
            math_lines = [stripped[2:]]; i += 1
            while i < len(lines) and '$$' not in lines[i]:
                math_lines.append(lines[i]); i += 1
            if i < len(lines):
                ep = lines[i].find('$$'); math_lines.append(lines[i][:ep])
            out.append(convert_math('\n'.join(math_lines), is_display=True)); i += 1; continue
        # \[...\] display math – three forms mirroring the $$ cases above
        # Form 1: lone \[ on its own line, collect until lone \]
        if stripped == r'\[':
            math_lines = []; i += 1
            while i < len(lines) and lines[i].strip() != r'\]':
                math_lines.append(lines[i]); i += 1
            out.append(convert_math('\n'.join(math_lines), is_display=True)); i += 1; continue
        # Form 2: \[...\] all on one line
        m = re.match(r'^\s*\\\[(.*?)\\\]\s*$', line, re.DOTALL)
        if m and m.group(1).strip():
            out.append(convert_math(m.group(1), is_display=True)); i += 1; continue
        # Form 3: \[ starts the line but \] has not appeared yet → collect until \]
        if stripped.startswith(r'\[') and r'\]' not in stripped:
            math_lines = [stripped[2:]]; i += 1
            while i < len(lines) and r'\]' not in lines[i]:
                math_lines.append(lines[i]); i += 1
            if i < len(lines):
                ep = lines[i].find(r'\]'); math_lines.append(lines[i][:ep])
            out.append(convert_math('\n'.join(math_lines), is_display=True)); i += 1; continue
        if not stripped:
            out.append(''); i += 1; continue
        if re.match(r'^[-*_]{3,}\s*$', line):
            out.append('#line(length: 100%)'); i += 1; continue
        out.append(convert_text_line(line)); i += 1
    return '\n'.join(out)

def main():
    parser = argparse.ArgumentParser(description='Convert Markdown + LaTeX math to Typst (AST version)')
    parser.add_argument('input', nargs='?', help='Input .md file (default: stdin)')
    parser.add_argument('-o', '--output',   help='Output .typ file (default: stdout)')
    args = parser.parse_args()
    src = open(args.input, encoding='utf-8').read() if args.input else sys.stdin.read()
    result = convert_document(src)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f: f.write(result)
        print(f'Wrote {args.output}', file=sys.stderr)
    else:
        print(result)

if __name__ == '__main__':
    main()