import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
 
fig, ax = plt.subplots(1, 1, figsize=(22, 30))
ax.set_xlim(0, 22)
ax.set_ylim(0, 30)
ax.axis('off')
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#0D1117')
 
# ── Color palette ──────────────────────────────────────────────
C = {
    'start':     '#F0F6FC',   # near-white
    'start_bd':  '#58A6FF',
    'conv':      '#1C4E80',   # deep blue
    'conv_bd':   '#58A6FF',
    'router':    '#0D5C2E',   # deep green
    'router_bd': '#3FB950',
    'vector':    '#0A3D62',   # slate blue
    'vector_bd': '#58A6FF',
    'graph':     '#0A3D62',
    'graph_bd':  '#58A6FF',
    'chroma':    '#1A1A4E',   # dark indigo
    'chroma_bd': '#8B949E',
    'neo4j':     '#1A1A4E',
    'neo4j_bd':  '#8B949E',
    'ctx':       '#2D2D2D',
    'ctx_bd':    '#8B949E',
    'analyst':   '#3D1A78',   # deep purple
    'analyst_bd':'#A371F7',
    'reviewer':  '#7A0000',   # deep red
    'reviewer_bd':'#F85149',
    'outputs':   '#1A3A1A',
    'outputs_bd':'#3FB950',
    'playbook':  '#4A3000',
    'playbook_bd':'#D29922',
    'formatter': '#003333',
    'fmt_bd':    '#39D0D0',
    'final':     '#F0F6FC',
    'final_bd':  '#58A6FF',
    'loop_bg':   '#161B22',
    'loop_bd':   '#58A6FF',
    'txt_light': '#F0F6FC',
    'txt_dark':  '#0D1117',
    'txt_muted': '#8B949E',
    'arrow':     '#8B949E',
    'arrow_hi':  '#58A6FF',
}
 
def box(ax, x, y, w, h, label, sublabel=None,
        fc='#1C4E80', ec='#58A6FF', tc='#F0F6FC',
        shape='rect', fontsize=12.5, bold=False):
    """Draw a rounded rectangle node."""
    pad = 0.18
    if shape == 'diamond':
        # Draw diamond
        cx, cy = x + w/2, y + h/2
        dx, dy = w/2, h/2
        diamond = plt.Polygon(
            [[cx, cy+dy],[cx+dx, cy],[cx, cy-dy],[cx-dx, cy]],
            closed=True, facecolor=fc, edgecolor=ec, linewidth=2.2, zorder=3)
        ax.add_patch(diamond)
        ax.text(cx, cy+(0.08 if sublabel else 0), label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold' if bold else 'normal',
                color=tc, zorder=4,
                fontfamily='DejaVu Sans')
        if sublabel:
            ax.text(cx, cy-0.35, sublabel, ha='center', va='center',
                    fontsize=10, color='#8B949E', zorder=4)
        return
 
    if shape == 'stadium':
        bp = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.15,rounding_size=0.45",
                            facecolor=fc, edgecolor=ec, linewidth=2.5, zorder=3)
    elif shape == 'cylinder':
        bp = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.12,rounding_size=0.22",
                            facecolor=fc, edgecolor=ec, linewidth=2.2,
                            linestyle='dashed', zorder=3)
    else:
        bp = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.12,rounding_size=0.25",
                            facecolor=fc, edgecolor=ec, linewidth=2.2, zorder=3)
    ax.add_patch(bp)
 
    ty = y + h/2 + (0.14 if sublabel else 0)
    ax.text(x + w/2, ty, label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal',
            color=tc, zorder=4, fontfamily='DejaVu Sans')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.32, sublabel,
                ha='center', va='center', fontsize=10,
                color='#8B949E', zorder=4)
 
def arr(ax, x1, y1, x2, y2, label='', color='#8B949E',
        lw=2.0, style='->', labelside='right', arrowstyle=None):
    """Draw an arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=arrowstyle or '-|>',
                    color=color, lw=lw,
                    connectionstyle='arc3,rad=0.0',
                    mutation_scale=18),
                zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        offset = 0.28 if labelside == 'right' else -0.28
        ax.text(mx + (offset if abs(x2-x1) < abs(y2-y1) else 0),
                my + (0 if abs(x2-x1) < abs(y2-y1) else 0.22),
                label, ha='center', va='center',
                fontsize=10, color='#D29922',
                fontfamily='DejaVu Sans', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='#0D1117', ec='none', alpha=0.85))
 
def curved_arr(ax, x1, y1, x2, y2, label='', color='#8B949E', rad=0.25, lw=2.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=color, lw=lw,
                    connectionstyle=f'arc3,rad={rad}',
                    mutation_scale=18),
                zorder=5)
    if label:
        mx = (x1+x2)/2 + rad*0.8
        my = (y1+y2)/2
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=10, color='#D29922',
                fontfamily='DejaVu Sans', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='#0D1117', ec='none', alpha=0.85))
 
# ══════════════════════════════════════════════════════════════
# LAYOUT  (x, y = bottom-left corner; w, h = size)
# ══════════════════════════════════════════════════════════════
# Y positions from top to bottom
Y_START    = 27.8
Y_CONV     = 25.6
Y_ROUTER   = 23.0          # diamond center y ≈ 23.7
Y_RAG_TOP  = 20.6          # Vector / Graph top edge
Y_CHROMA   = 18.5          # Chroma / Neo4j top
Y_CTX      = 16.5          # Context top
Y_ANALYST  = 14.2          # Analyst top
Y_REVIEWER = 11.8          # Reviewer top
Y_OUTPUTS  = 9.4
Y_PLAYBOOK = 7.2
Y_FORMAT   = 5.0
Y_FINAL    = 2.8
 
NW = 4.2    # node width std
NH = 1.0    # node height std
CX = 11.0   # center X of diagram
 
# Column X centers  (node left edge = COL_X - NW/2)
LEFT_X   = 5.8    # Vector / Chroma column center
RIGHT_X  = 16.2   # Graph / Neo4j column center
 
# ── Loop background ────────────────────────────────────────────
loop_rect = FancyBboxPatch((1.5, Y_REVIEWER - 0.5), 19.0, Y_ROUTER - Y_REVIEWER + 2.5,
                            boxstyle="round,pad=0.2,rounding_size=0.4",
                            facecolor=C['loop_bg'], edgecolor=C['loop_bd'],
                            linewidth=2.0, linestyle=(0,(8,4)), zorder=1, alpha=0.7)
ax.add_patch(loop_rect)
ax.text(2.2, Y_ROUTER + 1.8, '⟳  For each hypothesis  ×17',
        ha='left', va='center', fontsize=14, color=C['loop_bd'],
        fontfamily='DejaVu Sans', fontweight='bold', zorder=2)
 
# ── Retrieval Pipeline sub-background ────────────────────────
ret_rect = FancyBboxPatch((2.0, Y_CTX - 0.3), 18.0, Y_RAG_TOP - Y_CTX + 1.6,
                           boxstyle="round,pad=0.15,rounding_size=0.3",
                           facecolor='#111820', edgecolor='#2A4060',
                           linewidth=1.5, linestyle=(0,(5,5)), zorder=1, alpha=0.6)
ax.add_patch(ret_rect)
ax.text(11, Y_RAG_TOP + 1.28, 'Retrieval Pipeline',
        ha='center', va='center', fontsize=11.5, color='#2A6080',
        fontfamily='DejaVu Sans', fontstyle='italic', zorder=2)
 
# ── Agent Logic sub-background ────────────────────────────────
agt_rect = FancyBboxPatch((5.8, Y_REVIEWER - 0.35), 10.4, Y_ANALYST - Y_REVIEWER + 1.6,
                           boxstyle="round,pad=0.15,rounding_size=0.3",
                           facecolor='#1A0D22', edgecolor='#5A2080',
                           linewidth=1.5, linestyle=(0,(5,5)), zorder=1, alpha=0.6)
ax.add_patch(agt_rect)
ax.text(11, Y_ANALYST + 1.28, 'Agentic Review',
        ha='center', va='center', fontsize=11.5, color='#8050B0',
        fontfamily='DejaVu Sans', fontstyle='italic', zorder=2)
 
# ══════════════════════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════════════════════
# START
box(ax, CX-NW/2, Y_START, NW, NH,
    'Contract + User Prompt', '+ Conversation History',
    fc='#1A2840', ec=C['start_bd'], tc=C['txt_light'],
    shape='stadium', fontsize=12.5, bold=True)
 
# Conversation Agent
box(ax, CX-NW/2, Y_CONV, NW, NH+0.1,
    'Conversation Agent', 'Routes request, manages history',
    fc=C['conv'], ec=C['conv_bd'], tc=C['txt_light'],
    fontsize=12.5, bold=True)
 
# Router (diamond)
RDW, RDH = 4.8, 2.0   # diamond bounding box
box(ax, CX-RDW/2, Y_ROUTER, RDW, RDH,
    'Retrieval Router', 'vector RAG or GraphRAG',
    fc=C['router'], ec=C['router_bd'], tc='#7EE787',
    shape='diamond', fontsize=12.5, bold=True)
 
# Vector RAG
VX = LEFT_X - NW/2
box(ax, VX, Y_RAG_TOP, NW, NH,
    'Vector RAG Pipeline', 'Embedding + cosine similarity',
    fc=C['vector'], ec=C['vector_bd'], tc=C['txt_light'],
    fontsize=11.5, bold=True)
 
# Graph RAG
GX = RIGHT_X - NW/2
box(ax, GX, Y_RAG_TOP, NW, NH,
    'GraphRAG Pipeline', 'Knowledge graph retrieval',
    fc=C['graph'], ec=C['graph_bd'], tc=C['txt_light'],
    fontsize=11.5, bold=True)
 
# Chroma DB
box(ax, VX, Y_CHROMA, NW, NH,
    '⬡  Chroma', 'Vector Database',
    fc=C['chroma'], ec=C['chroma_bd'], tc='#A0AFBF',
    shape='cylinder', fontsize=12, bold=False)
 
# Neo4j DB
box(ax, GX, Y_CHROMA, NW, NH,
    '⬡  Neo4j', 'Graph Database',
    fc=C['neo4j'], ec=C['neo4j_bd'], tc='#A0AFBF',
    shape='cylinder', fontsize=12, bold=False)
 
# Context
box(ax, CX-NW/2-0.3, Y_CTX, NW+0.6, NH,
    'Retrieved Context',
    fc='#1E2830', ec='#4A6070', tc='#B0C8D8',
    fontsize=12.5, bold=True)
 
# Analyst
box(ax, CX-NW/2-0.4, Y_ANALYST, NW+0.8, NH,
    'Hypothesis Analyst', 'Answers one hypothesis',
    fc=C['analyst'], ec=C['analyst_bd'], tc='#D2B4FF',
    fontsize=12.5, bold=True)
 
# Reviewer
box(ax, CX-NW/2-0.4, Y_REVIEWER, NW+0.8, NH,
    'Reviewer Agent', 'Validates answer quality',
    fc=C['reviewer'], ec=C['reviewer_bd'], tc='#FFB3B0',
    fontsize=12.5, bold=True)
 
# Outputs
OW = NW + 1.0
box(ax, CX-OW/2, Y_OUTPUTS, OW, NH,
    '17 Validated Verdicts + Evidence Spans',
    fc=C['outputs'], ec=C['outputs_bd'], tc='#7EE787',
    fontsize=12, bold=True)
 
# Playbook
box(ax, CX-NW/2, Y_PLAYBOOK, NW, NH,
    'Playbook Enrichment', 'Policy context, risk flags, actions',
    fc=C['playbook'], ec=C['playbook_bd'], tc='#F0C060',
    fontsize=12, bold=True)
 
# Formatter
box(ax, CX-NW/2, Y_FORMAT, NW, NH,
    'Runtrace Formatter', 'Structures all outputs per schema',
    fc=C['formatter'], ec=C['fmt_bd'], tc='#50E8E8',
    fontsize=12, bold=True)
 
# Final
box(ax, CX-NW/2, Y_FINAL, NW, NH,
    'Final NDA Review Output',
    fc='#1A2840', ec='#58A6FF', tc='#F0F6FC',
    shape='stadium', fontsize=12.5, bold=True)
 
# ══════════════════════════════════════════════════════════════
# ARROWS
# ══════════════════════════════════════════════════════════════
# Start → ConvAgent
arr(ax, CX, Y_START, CX, Y_CONV+NH+0.1,
    'contract + prompt + history', color=C['arrow_hi'], lw=2.2)
 
# ConvAgent → Router
arr(ax, CX, Y_CONV, CX, Y_ROUTER+RDH,
    'parsed query + session context', color=C['arrow_hi'], lw=2.2)
 
# Router → Vector (left diagonal)
arr(ax, CX - RDW/2, Y_ROUTER + RDH/2,
    VX + NW, Y_RAG_TOP + NH/2,
    'query embedding', color='#3FB950', lw=2.0, labelside='left')
 
# Router → Graph (right diagonal)
arr(ax, CX + RDW/2, Y_ROUTER + RDH/2,
    GX, Y_RAG_TOP + NH/2,
    'graph query', color='#3FB950', lw=2.0, labelside='right')
 
# ── Vector Pipeline → Chroma  (straight down, offset right so return is visible)
arr(ax, LEFT_X + 0.2, Y_RAG_TOP,
        LEFT_X + 0.2, Y_CHROMA + NH,
    '', color='#4080C0', lw=1.8)
 
# ── Chroma → Vector Pipeline  (straight up, offset left)
ax.annotate('', xy=(LEFT_X - 0.2, Y_RAG_TOP),
            xytext=(LEFT_X - 0.2, Y_CHROMA + NH),
            arrowprops=dict(arrowstyle='-|>', color='#2060A0', lw=1.4,
                            connectionstyle='arc3,rad=0.0', mutation_scale=14),
            zorder=5)
 
# ── Graph Pipeline → Neo4j  (straight down, offset left)
arr(ax, RIGHT_X - 0.2, Y_RAG_TOP,
        RIGHT_X - 0.2, Y_CHROMA + NH,
    '', color='#4080C0', lw=1.8)
 
# ── Neo4j → Graph Pipeline  (straight up, offset right)
ax.annotate('', xy=(RIGHT_X + 0.2, Y_RAG_TOP),
            xytext=(RIGHT_X + 0.2, Y_CHROMA + NH),
            arrowprops=dict(arrowstyle='-|>', color='#2060A0', lw=1.4,
                            connectionstyle='arc3,rad=0.0', mutation_scale=14),
            zorder=5)
 
# ── Vector RAG → Retrieved Context
# Route: Inner side of Vector RAG → midpoint → Retrieved Context
CTX_W   = NW + 0.6
CTX_LEFT_X  = CX - CTX_W / 2
CTX_RIGHT_X = CX + CTX_W / 2
CTX_MID_Y   = Y_CTX + NH / 2

# Path for Vector RAG (Inner side)
ax.annotate('', xy=(CTX_LEFT_X, CTX_MID_Y),
            xytext=(LEFT_X + NW/2, Y_RAG_TOP + NH/2),
            arrowprops=dict(arrowstyle='-|>', color='#58A6FF', lw=1.8,
                            connectionstyle='angle,angleA=0,angleB=90,rad=10',
                            mutation_scale=16),
            zorder=5)
 
# ── Graph RAG → Retrieved Context
# Path for Graph RAG (Inner side)
ax.annotate('', xy=(CTX_RIGHT_X, CTX_MID_Y),
            xytext=(RIGHT_X - NW/2, Y_RAG_TOP + NH/2),
            arrowprops=dict(arrowstyle='-|>', color='#58A6FF', lw=1.8,
                            connectionstyle='angle,angleA=0,angleB=90,rad=10',
                            mutation_scale=16),
            zorder=5)
 
# Context → Analyst
arr(ax, CX, Y_CTX, CX, Y_ANALYST+NH,
    'context + contract + hypothesis', color='#8B60D0', lw=2.0)
 
# Analyst → Reviewer
arr(ax, CX, Y_ANALYST, CX, Y_REVIEWER+NH,
    'verdict + evidence', color='#A371F7', lw=2.0)
 
# Reviewer → Analyst (curved retry, on right side)
curved_arr(ax, CX+NW/2+0.0, Y_REVIEWER+NH/2,
           CX+NW/2+0.0, Y_ANALYST+NH/2,
           label='Retry if rejected\n(max 3 tries)',
           color='#F85149', rad=-0.55, lw=2.0)
 
# Reviewer → Outputs
arr(ax, CX, Y_REVIEWER, CX, Y_OUTPUTS+NH,
    '17 validated verdicts + evidence spans', color='#3FB950', lw=2.2)
 
# Outputs → Playbook
arr(ax, CX, Y_OUTPUTS, CX, Y_PLAYBOOK+NH,
    'verdicts + evidence spans', color=C['playbook_bd'], lw=2.0)
 
# Playbook → Formatter
arr(ax, CX, Y_PLAYBOOK, CX, Y_FORMAT+NH,
    'enriched verdicts + evidence spans', color=C['fmt_bd'], lw=2.0)
 
# Formatter → Final
arr(ax, CX, Y_FORMAT, CX, Y_FINAL+NH,
    'structured JSON runtrace', color=C['start_bd'], lw=2.2)
 
# ── Title ──────────────────────────────────────────────────────
ax.text(11, 29.5, 'NDA Review Pipeline — Architecture',
        ha='center', va='center', fontsize=20, fontweight='bold',
        color='#F0F6FC', fontfamily='DejaVu Sans', zorder=10)
ax.text(11, 29.15, 'Multi-agent retrieval-augmented contract analysis system',
        ha='center', va='center', fontsize=13, color='#8B949E',
        fontfamily='DejaVu Sans', zorder=10)
 
# ── Legend ────────────────────────────────────────────────────
legend_items = [
    (C['conv'],     C['conv_bd'],     'Orchestration / Routing'),
    (C['vector'],   C['vector_bd'],   'RAG Retrieval (Vector / Graph)'),
    (C['chroma'],   C['chroma_bd'],   'Databases  (Chroma / Neo4j)'),
    (C['analyst'],  C['analyst_bd'],  'Analysis Agent'),
    (C['reviewer'], C['reviewer_bd'], 'Review Agent'),
    (C['playbook'], C['playbook_bd'], 'Enrichment'),
    (C['formatter'],C['fmt_bd'],      'Formatting & Output'),
]
lx, ly = 15.8, 4.2
for fc, ec, lbl in legend_items:
    bp = FancyBboxPatch((lx, ly), 0.55, 0.38,
                        boxstyle="round,pad=0.05,rounding_size=0.08",
                        facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=8)
    ax.add_patch(bp)
    ax.text(lx + 0.72, ly + 0.19, lbl, va='center',
            fontsize=10.5, color='#C8D8E8',
            fontfamily='DejaVu Sans', zorder=9)
    ly -= 0.55
 
plt.tight_layout(pad=0.3)
plt.savefig('nda_pipeline.pdf',
            facecolor='#0D1117', bbox_inches='tight', dpi=200)
plt.savefig('nda_pipeline.png',
            facecolor='#0D1117', bbox_inches='tight', dpi=150)
print("Done")