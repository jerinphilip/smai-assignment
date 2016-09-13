import numpy as np
import sys

def toLatexMatrix(W):
    col = lambda x: ' & '.join(list(map(lambda y: "%0.2f"%(y), x)))
    rows = '\\\\\n'.join(list(map(col, W)))
    total = '\\begin{bmatrix}\n' + rows + '\n\\end{bmatrix}'
    return total
    #print(W)

def toLatexTable(W):
    m, n = W.shape
    start = '\\begin{tabular}{'+'c'.join((['|']*(n+1)))+'}'
    hline = '\\hline '
    nline = '\\\\ '
    cols = lambda x: ' & '.join(list(map(lambda y: "%0.2f"%(y), x)))
    rows = (nline + hline + '\n').join(list(map(cols, W)))
    end = '\\end{tabular}'
    lines = [start, hline, rows+nline, hline, end]
    output = '\n'.join(lines)
    return output
    
    

def toEquation(packed):
    i, W = packed
    m, n = W.shape
    mname = "W_{%d%d}"%(i+2, i+1)
    if m < n:
        W = W.T
        lW = W.tolist()
        mname += '^\\mathsf{T}'
    equation = "\\begin{equation*} \n%s = %s \n\\end{equation*}"%(mname, 
            toLatexMatrix(W))
    return equation

def toSubSection(packed):
    i, W = packed
    m, n = W.shape
    mname = "W_{%d%d}"%(i+2, i+1)
    if m < n:
        W = W.T
        lW = W.tolist()
        mname += '^\\mathsf{T}'
    subsection = '\\subsubsection{%s}\n'%(mname)
    content = toLatexTable(W)
    return subsection + '\n' + content

def extract_neurons(WS):
    ns = list(map(lambda x: x.shape[0], WS))
    return [WS[0].shape[1]] + ns


WS = np.load(sys.argv[1])
ns = extract_neurons(WS)
equations = map(toEquation, zip(range(len(WS)), WS))
subsections = map(toSubSection, zip(range(len(WS)), WS))

ns_str = 'Neurons ['+ ' '.join(map(str, ns))+']\n\n'
print(ns_str+'\n'.join(subsections))

