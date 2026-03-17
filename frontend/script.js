/* ============================================================
   VQML Benchmark — Interactive Dashboard Script
   ============================================================ */

// ============================================================
// BENCHMARK DATA (representative results)
// ============================================================
const ALGO_DATA = {
    qnn:  { name: 'QNN',  fullName: 'Quantum Neural Network',                accuracy: 0.528, precision: 0.528, recall: 1.000, f1: 0.691 },
    qcnn: { name: 'QCNN', fullName: 'Quantum Convolutional Neural Network', accuracy: 0.964, precision: 0.942, recall: 0.992, f1: 0.966 },
    vqc:  { name: 'VQC',  fullName: 'Variational Quantum Classifier',       accuracy: 0.964, precision: 0.942, recall: 0.992, f1: 0.966 },
    vqfe: { name: 'VQFE', fullName: 'Variational Quantum Feature Embedding',accuracy: 0.576, precision: 0.591, recall: 0.636, f1: 0.613 },
    qsvm: { name: 'QSVM', fullName: 'Quantum Support Vector Machine',      accuracy: 0.900, precision: 0.928, recall: 0.928, f1: 0.928 },
};

const METRICS = ['accuracy', 'precision', 'recall', 'f1'];
const METRIC_COLORS = {
    accuracy:  '#5b7fa6',
    precision: '#8e7ab5',
    recall:    '#c47a8e',
    f1:        '#6aab8e'
};
const METRIC_LABELS = {
    accuracy: 'Accuracy',
    precision: 'Precision',
    recall: 'Recall',
    f1: 'F1-Score'
};

// ============================================================
// QUANTUM PARTICLE CANVAS ANIMATION
// ============================================================
(function initQuantumCanvas() {
    const canvas = document.getElementById('quantum-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let w, h, particles = [], connections = [];
    const PARTICLE_COUNT = 40;
    const MAX_DIST = 120;

    function resize() {
        w = canvas.width = window.innerWidth;
        h = canvas.height = window.innerHeight;
    }

    class Particle {
        constructor() {
            this.x = Math.random() * w;
            this.y = Math.random() * h;
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.r = Math.random() * 2 + 0.5;
            this.hue = Math.random() > 0.5 ? 215 : 260;
            this.alpha = Math.random() * 0.2 + 0.08;
        }
        update() {
            this.x += this.vx;
            this.y += this.vy;
            if (this.x < 0) this.x = w;
            if (this.x > w) this.x = 0;
            if (this.y < 0) this.y = h;
            if (this.y > h) this.y = 0;
        }
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${this.hue}, 25%, 55%, ${this.alpha})`;
            ctx.fill();
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.r * 2.5, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${this.hue}, 25%, 55%, ${this.alpha * 0.1})`;
            ctx.fill();
        }
    }

    function init() {
        resize();
        particles = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            particles.push(new Particle());
        }
    }

    function animate() {
        ctx.clearRect(0, 0, w, h);
        particles.forEach(p => {
            p.update();
            p.draw();
        });
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < MAX_DIST) {
                    const alpha = (1 - dist / MAX_DIST) * 0.15;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `hsla(215, 25%, 55%, ${alpha})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animate);
    }

    window.addEventListener('resize', resize);
    init();
    animate();
})();


// ============================================================
// NAVIGATION
// ============================================================
(function initNav() {
    const nav = document.getElementById('main-nav');
    const toggle = document.getElementById('nav-toggle');
    const links = document.querySelector('.nav-links');
    const navAs = document.querySelectorAll('.nav-links a');

    window.addEventListener('scroll', () => {
        nav.classList.toggle('scrolled', window.scrollY > 50);
    });

    toggle.addEventListener('click', () => {
        links.classList.toggle('open');
    });

    navAs.forEach(a => {
        a.addEventListener('click', () => {
            links.classList.remove('open');
        });
    });

    const sections = document.querySelectorAll('section[id]');
    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(s => {
            const top = s.offsetTop - 120;
            if (window.scrollY >= top) current = s.id;
        });
        navAs.forEach(a => {
            a.classList.toggle('active', a.getAttribute('href') === '#' + current);
        });
    });
})();


// ============================================================
// SCROLL ANIMATIONS (IntersectionObserver)
// ============================================================
(function initScrollAnimations() {
    const items = document.querySelectorAll('.anim-slide-up');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, i) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, i * 80);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

    items.forEach(el => observer.observe(el));
})();


// ============================================================
// ALGORITHM CARDS — Expand/Collapse
// ============================================================
(function initAlgoCards() {
    const cards = document.querySelectorAll('.algo-card');
    cards.forEach(card => {
        card.addEventListener('click', () => {
            const wasExpanded = card.classList.contains('expanded');
            cards.forEach(c => c.classList.remove('expanded'));
            if (!wasExpanded) card.classList.add('expanded');
        });
    });
})();


// ============================================================
// CIRCUIT DIAGRAMS (SVG-based)
// ============================================================
(function initCircuitDiagrams() {
    const circuits = {
        qnn: {
            qubits: 4,
            gates: [
                { type: 'angle', qubits: [0,1,2,3], label: 'AngleEmb' },
                { type: 'entangle', qubits: [0,1,2,3], label: 'BasicEnt ×3' },
            ],
            measure: 0
        },
        qcnn: {
            qubits: 4,
            gates: [
                { type: 'angle', qubits: [0,1,2,3], label: 'AngleEmb' },
                { type: 'conv', qubits: [0,1], label: 'RX/RY' },
                { type: 'conv', qubits: [2,3], label: 'RX/RY' },
                { type: 'pool', qubits: [0], label: 'Rot' },
            ],
            measure: 0
        },
        vqc: {
            qubits: 4,
            gates: [
                { type: 'angle', qubits: [0,1,2,3], label: 'AngleEmb' },
                { type: 'strong', qubits: [0,1,2,3], label: 'StrongEnt ×3' },
            ],
            measure: 0
        },
        vqfe: {
            qubits: 4,
            gates: [
                { type: 'trainable', qubits: [0,1,2,3], label: 'RY(x·w)' },
                { type: 'entangle', qubits: [0,1,2,3], label: 'BasicEnt ×3' },
            ],
            measure: 0
        },
        qsvm: {
            qubits: 4,
            gates: [
                { type: 'angle', qubits: [0,1,2,3], label: 'AngleEmb(x₁)' },
                { type: 'adjoint', qubits: [0,1,2,3], label: 'AngleEmb†(x₂)' },
            ],
            measure: -1
        }
    };

    const gateColors = {
        angle:     { fill: '#5b7fa618', stroke: '#5b7fa6', text: '#5b7fa6' },
        entangle:  { fill: '#8e7ab518', stroke: '#8e7ab5', text: '#8e7ab5' },
        strong:    { fill: '#c47a8e18', stroke: '#c47a8e', text: '#c47a8e' },
        conv:      { fill: '#c9a95a18', stroke: '#c9a95a', text: '#c9a95a' },
        pool:      { fill: '#6aab8e18', stroke: '#6aab8e', text: '#6aab8e' },
        trainable: { fill: '#c47a8e18', stroke: '#c47a8e', text: '#c47a8e' },
        adjoint:   { fill: '#8e7ab518', stroke: '#8e7ab5', text: '#8e7ab5' },
    };

    Object.entries(circuits).forEach(([key, circ]) => {
        const container = document.getElementById(`circuit-${key}`);
        if (!container) return;

        const svgNS = 'http://www.w3.org/2000/svg';
        const W = 420, H = 24 + circ.qubits * 28;
        const svg = document.createElementNS(svgNS, 'svg');
        svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
        svg.setAttribute('width', '100%');

        const wireY = (q) => 20 + q * 28;

        for (let q = 0; q < circ.qubits; q++) {
            const y = wireY(q);
            const lbl = document.createElementNS(svgNS, 'text');
            lbl.setAttribute('x', 8);
            lbl.setAttribute('y', y + 4);
            lbl.setAttribute('fill', '#9a9490');
            lbl.setAttribute('font-size', '10');
            lbl.setAttribute('font-family', "'JetBrains Mono', monospace");
            lbl.textContent = `q${q}`;
            svg.appendChild(lbl);

            const line = document.createElementNS(svgNS, 'line');
            line.setAttribute('x1', 30);
            line.setAttribute('y1', y);
            line.setAttribute('x2', W - 10);
            line.setAttribute('y2', y);
            line.setAttribute('stroke', '#9a949040');
            line.setAttribute('stroke-width', '1');
            svg.appendChild(line);
        }

        let gx = 50;
        circ.gates.forEach(gate => {
            const colors = gateColors[gate.type];
            const minQ = Math.min(...gate.qubits);
            const maxQ = Math.max(...gate.qubits);
            const gh = (maxQ - minQ) * 28 + 20;
            const gy = wireY(minQ) - 10;
            const gw = Math.max(gate.label.length * 7.5 + 16, 60);

            const rect = document.createElementNS(svgNS, 'rect');
            rect.setAttribute('x', gx);
            rect.setAttribute('y', gy);
            rect.setAttribute('width', gw);
            rect.setAttribute('height', gh);
            rect.setAttribute('rx', '5');
            rect.setAttribute('fill', colors.fill);
            rect.setAttribute('stroke', colors.stroke);
            rect.setAttribute('stroke-width', '1.5');
            svg.appendChild(rect);

            const txt = document.createElementNS(svgNS, 'text');
            txt.setAttribute('x', gx + gw / 2);
            txt.setAttribute('y', gy + gh / 2 + 4);
            txt.setAttribute('fill', colors.text);
            txt.setAttribute('font-size', '10');
            txt.setAttribute('font-weight', '600');
            txt.setAttribute('font-family', "'JetBrains Mono', monospace");
            txt.setAttribute('text-anchor', 'middle');
            txt.textContent = gate.label;
            svg.appendChild(txt);

            gx += gw + 16;
        });

        if (circ.measure >= 0) {
            const my = wireY(circ.measure);
            const mx = gx + 8;
            const mrect = document.createElementNS(svgNS, 'rect');
            mrect.setAttribute('x', mx);
            mrect.setAttribute('y', my - 10);
            mrect.setAttribute('width', 36);
            mrect.setAttribute('height', 20);
            mrect.setAttribute('rx', '4');
            mrect.setAttribute('fill', '#6aab8e18');
            mrect.setAttribute('stroke', '#6aab8e');
            mrect.setAttribute('stroke-width', '1.2');
            svg.appendChild(mrect);

            const mtxt = document.createElementNS(svgNS, 'text');
            mtxt.setAttribute('x', mx + 18);
            mtxt.setAttribute('y', my + 4);
            mtxt.setAttribute('fill', '#6aab8e');
            mtxt.setAttribute('font-size', '9');
            mtxt.setAttribute('font-weight', '700');
            mtxt.setAttribute('font-family', "'JetBrains Mono', monospace");
            mtxt.setAttribute('text-anchor', 'middle');
            mtxt.textContent = '⟨Z₀⟩';
            svg.appendChild(mtxt);
        } else {
            const my = wireY(0);
            const mx = gx + 8;
            const mrect = document.createElementNS(svgNS, 'rect');
            mrect.setAttribute('x', mx);
            mrect.setAttribute('y', my - 10);
            mrect.setAttribute('width', 44);
            mrect.setAttribute('height', 20 + (circ.qubits - 1) * 28);
            mrect.setAttribute('rx', '4');
            mrect.setAttribute('fill', '#6aab8e18');
            mrect.setAttribute('stroke', '#6aab8e');
            mrect.setAttribute('stroke-width', '1.2');
            svg.appendChild(mrect);

            const mtxt = document.createElementNS(svgNS, 'text');
            mtxt.setAttribute('x', mx + 22);
            mtxt.setAttribute('y', my + (circ.qubits - 1) * 14 + 4);
            mtxt.setAttribute('fill', '#6aab8e');
            mtxt.setAttribute('font-size', '9');
            mtxt.setAttribute('font-weight', '700');
            mtxt.setAttribute('font-family', "'JetBrains Mono', monospace");
            mtxt.setAttribute('text-anchor', 'middle');
            mtxt.textContent = 'Probs';
            svg.appendChild(mtxt);
        }

        container.appendChild(svg);
    });
})();


// ============================================================
// PERFORMANCE RING ANIMATIONS
// ============================================================
(function initPerformanceRings() {
    const circumference = 2 * Math.PI * 42; // r=42

    // Set ring colors from data attributes
    document.querySelectorAll('.ring-item').forEach(item => {
        const color = item.dataset.color;
        const fg = item.querySelector('.ring-fg');
        if (fg && color) {
            fg.style.stroke = color;
        }
    });

    // Animate rings on scroll
    const perfGrid = document.getElementById('perf-grid');
    if (!perfGrid) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const rings = entry.target.querySelectorAll('.ring-fg');
                rings.forEach((ring, i) => {
                    const value = parseInt(ring.dataset.value) || 0;
                    const offset = circumference - (value / 100) * circumference;
                    setTimeout(() => {
                        ring.style.strokeDashoffset = offset;
                    }, i * 100);
                });
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.2 });

    document.querySelectorAll('.perf-card').forEach(card => {
        observer.observe(card);
    });
})();


// ============================================================
// CHARTS — Bar, Radar, Table
// ============================================================
const chartViews = {
    bar: null,
    radar: null,
    table: null,
    rendered: { bar: false, radar: false, table: false }
};

function renderBarChart() {
    if (chartViews.rendered.bar) return;
    chartViews.rendered.bar = true;

    const container = document.getElementById('chart-bar');
    if (!container) return;

    const svgNS = 'http://www.w3.org/2000/svg';
    const algos = Object.keys(ALGO_DATA);
    const W = 800, H = 360;
    const padding = { top: 30, right: 20, bottom: 60, left: 50 };
    const chartW = W - padding.left - padding.right;
    const chartH = H - padding.top - padding.bottom;

    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('width', '100%');
    svg.style.maxHeight = '360px';

    for (let i = 0; i <= 5; i++) {
        const y = padding.top + (chartH / 5) * i;
        const line = document.createElementNS(svgNS, 'line');
        line.setAttribute('x1', padding.left);
        line.setAttribute('y1', y);
        line.setAttribute('x2', W - padding.right);
        line.setAttribute('y2', y);
        line.setAttribute('stroke', '#2d292608');
        line.setAttribute('stroke-width', '1');
        svg.appendChild(line);

        const lbl = document.createElementNS(svgNS, 'text');
        lbl.setAttribute('x', padding.left - 8);
        lbl.setAttribute('y', y + 4);
        lbl.setAttribute('fill', '#9a9490');
        lbl.setAttribute('font-size', '10');
        lbl.setAttribute('text-anchor', 'end');
        lbl.setAttribute('font-family', "'JetBrains Mono', monospace");
        lbl.textContent = (100 - i * 20) + '%';
        svg.appendChild(lbl);
    }

    const groupW = chartW / algos.length;
    const barW = (groupW - 20) / 4;

    algos.forEach((algoKey, gi) => {
        const algo = ALGO_DATA[algoKey];
        const gx = padding.left + gi * groupW + 10;

        METRICS.forEach((metric, mi) => {
            const val = algo[metric];
            const bx = gx + mi * barW;
            const barH = val * chartH;
            const by = padding.top + chartH - barH;

            const rect = document.createElementNS(svgNS, 'rect');
            rect.setAttribute('x', bx + 1);
            rect.setAttribute('y', by);
            rect.setAttribute('width', barW - 2);
            rect.setAttribute('height', barH);
            rect.setAttribute('rx', '3');
            rect.setAttribute('fill', METRIC_COLORS[metric]);
            rect.setAttribute('opacity', '0.85');
            rect.style.transition = 'opacity 0.2s';

            const title = document.createElementNS(svgNS, 'title');
            title.textContent = `${algo.name} — ${METRIC_LABELS[metric]}: ${(val * 100).toFixed(1)}%`;
            rect.appendChild(title);

            rect.addEventListener('mouseenter', () => rect.setAttribute('opacity', '1'));
            rect.addEventListener('mouseleave', () => rect.setAttribute('opacity', '0.85'));

            svg.appendChild(rect);
        });

        const lbl = document.createElementNS(svgNS, 'text');
        lbl.setAttribute('x', gx + (groupW - 10) / 2);
        lbl.setAttribute('y', H - 20);
        lbl.setAttribute('fill', '#6b6560');
        lbl.setAttribute('font-size', '12');
        lbl.setAttribute('font-weight', '700');
        lbl.setAttribute('text-anchor', 'middle');
        lbl.setAttribute('font-family', "'JetBrains Mono', monospace");
        lbl.textContent = algo.name;
        svg.appendChild(lbl);
    });

    container.appendChild(svg);
}

function renderRadarChart() {
    if (chartViews.rendered.radar) return;
    chartViews.rendered.radar = true;

    const container = document.getElementById('chart-radar');
    if (!container) return;

    const svgNS = 'http://www.w3.org/2000/svg';
    const W = 500, H = 420;
    const cx = W / 2, cy = H / 2 - 10;
    const R = 150;
    const levels = 5;

    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('width', '100%');
    svg.style.maxHeight = '420px';

    const angleStep = (Math.PI * 2) / METRICS.length;

    for (let lv = 1; lv <= levels; lv++) {
        const r = (R / levels) * lv;
        let points = '';
        for (let i = 0; i < METRICS.length; i++) {
            const a = -Math.PI / 2 + i * angleStep;
            points += `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)} `;
        }
        const poly = document.createElementNS(svgNS, 'polygon');
        poly.setAttribute('points', points.trim());
        poly.setAttribute('fill', 'none');
        poly.setAttribute('stroke', '#2d292608');
        poly.setAttribute('stroke-width', '1');
        svg.appendChild(poly);
    }

    METRICS.forEach((metric, i) => {
        const a = -Math.PI / 2 + i * angleStep;
        const lx = cx + (R + 24) * Math.cos(a);
        const ly = cy + (R + 24) * Math.sin(a);

        const line = document.createElementNS(svgNS, 'line');
        line.setAttribute('x1', cx);
        line.setAttribute('y1', cy);
        line.setAttribute('x2', cx + R * Math.cos(a));
        line.setAttribute('y2', cy + R * Math.sin(a));
        line.setAttribute('stroke', '#2d29260a');
        svg.appendChild(line);

        const lbl = document.createElementNS(svgNS, 'text');
        lbl.setAttribute('x', lx);
        lbl.setAttribute('y', ly + 4);
        lbl.setAttribute('fill', METRIC_COLORS[metric]);
        lbl.setAttribute('font-size', '11');
        lbl.setAttribute('font-weight', '600');
        lbl.setAttribute('text-anchor', 'middle');
        lbl.setAttribute('font-family', "'JetBrains Mono', monospace");
        lbl.textContent = METRIC_LABELS[metric];
        svg.appendChild(lbl);
    });

    const algoColors = {
        qnn:  { stroke: '#c47a8e', fill: '#c47a8e20' },
        qcnn: { stroke: '#5b7fa6', fill: '#5b7fa618' },
        vqc:  { stroke: '#8e7ab5', fill: '#8e7ab518' },
        vqfe: { stroke: '#c9a95a', fill: '#c9a95a18' },
        qsvm: { stroke: '#6aab8e', fill: '#6aab8e18' },
    };

    Object.entries(ALGO_DATA).forEach(([key, algo]) => {
        let points = '';
        METRICS.forEach((metric, i) => {
            const a = -Math.PI / 2 + i * angleStep;
            const r = algo[metric] * R;
            points += `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)} `;
        });

        const poly = document.createElementNS(svgNS, 'polygon');
        poly.setAttribute('points', points.trim());
        poly.setAttribute('fill', algoColors[key].fill);
        poly.setAttribute('stroke', algoColors[key].stroke);
        poly.setAttribute('stroke-width', '2');
        poly.style.transition = 'opacity 0.2s';
        svg.appendChild(poly);

        METRICS.forEach((metric, i) => {
            const a = -Math.PI / 2 + i * angleStep;
            const r = algo[metric] * R;
            const dot = document.createElementNS(svgNS, 'circle');
            dot.setAttribute('cx', cx + r * Math.cos(a));
            dot.setAttribute('cy', cy + r * Math.sin(a));
            dot.setAttribute('r', '3');
            dot.setAttribute('fill', algoColors[key].stroke);

            const title = document.createElementNS(svgNS, 'title');
            title.textContent = `${algo.name} — ${METRIC_LABELS[metric]}: ${(algo[metric] * 100).toFixed(1)}%`;
            dot.appendChild(title);

            svg.appendChild(dot);
        });
    });

    let ly = H - 20;
    let lx = 60;
    Object.entries(ALGO_DATA).forEach(([key, algo]) => {
        const dot = document.createElementNS(svgNS, 'circle');
        dot.setAttribute('cx', lx);
        dot.setAttribute('cy', ly - 3);
        dot.setAttribute('r', '5');
        dot.setAttribute('fill', algoColors[key].stroke);
        svg.appendChild(dot);

        const lbl = document.createElementNS(svgNS, 'text');
        lbl.setAttribute('x', lx + 12);
        lbl.setAttribute('y', ly);
        lbl.setAttribute('fill', '#6b6560');
        lbl.setAttribute('font-size', '11');
        lbl.setAttribute('font-weight', '600');
        lbl.setAttribute('font-family', "'JetBrains Mono', monospace");
        lbl.textContent = algo.name;
        svg.appendChild(lbl);

        lx += 80;
    });

    container.appendChild(svg);
}

function renderDataTable() {
    if (chartViews.rendered.table) return;
    chartViews.rendered.table = true;

    const container = document.getElementById('chart-table');
    if (!container) return;

    const bests = {};
    METRICS.forEach(m => {
        let maxVal = -1;
        Object.values(ALGO_DATA).forEach(a => {
            if (a[m] > maxVal) maxVal = a[m];
        });
        bests[m] = maxVal;
    });

    let html = `<div class="data-table-wrap"><table class="data-table">
        <thead><tr>
            <th>Algorithm</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
        </tr></thead><tbody>`;

    Object.entries(ALGO_DATA).forEach(([key, algo]) => {
        html += `<tr>`;
        html += `<td class="algo-name-cell">${algo.name}</td>`;
        METRICS.forEach(m => {
            const val = algo[m];
            const isBest = val === bests[m];
            html += `<td class="metric-cell ${isBest ? 'best-cell' : ''}">${(val * 100).toFixed(1)}%</td>`;
        });
        html += `</tr>`;
    });

    html += `</tbody></table></div>`;
    container.innerHTML = html;
}

// Chart tab switching
(function initChartControls() {
    const buttons = document.querySelectorAll('.chart-btn');
    const containers = {
        bar: document.getElementById('chart-bar'),
        radar: document.getElementById('chart-radar'),
        table: document.getElementById('chart-table'),
    };

    renderBarChart();

    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const chart = btn.dataset.chart;
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            Object.entries(containers).forEach(([key, el]) => {
                el.classList.toggle('hidden', key !== chart);
            });

            if (chart === 'bar') renderBarChart();
            if (chart === 'radar') renderRadarChart();
            if (chart === 'table') renderDataTable();
        });
    });
})();


// ============================================================
// PREDICTION GALLERY
// ============================================================
(function initGallery() {
    const tabs = document.querySelectorAll('.gallery-tab');
    const img = document.getElementById('gallery-img');
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const lightboxClose = document.getElementById('lightbox-close');
    const display = document.querySelector('.gallery-display');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            const algo = tab.dataset.img;
            img.src = `../saved_models/${algo}_predictions.png`;
            img.alt = `${algo.toUpperCase()} Predictions on Test Set`;
        });
    });

    display.addEventListener('click', () => {
        lightboxImg.src = img.src;
        lightbox.classList.add('open');
    });

    lightboxClose.addEventListener('click', () => {
        lightbox.classList.remove('open');
    });

    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox) lightbox.classList.remove('open');
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') lightbox.classList.remove('open');
    });
})();


// ============================================================
// DRAW-A-DIGIT DEMO
// ============================================================
(function initDemoCanvas() {
    const canvas = document.getElementById('draw-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clear-canvas');
    const classifyBtn = document.getElementById('classify-btn');
    const resultArea = document.getElementById('demo-result');

    let drawing = false;
    let hasDrawn = false;

    function clearCanvas() {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        hasDrawn = false;
        resultArea.innerHTML = `
            <div class="result-placeholder">
                <div class="result-atom">⟨ψ|</div>
                <p>Draw a digit and click classify to see results</p>
            </div>`;
    }
    clearCanvas();

    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#2d2926';

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        if (e.touches) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY
            };
        }
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    function startDraw(e) {
        e.preventDefault();
        drawing = true;
        hasDrawn = true;
        const pos = getPos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    }

    function draw(e) {
        if (!drawing) return;
        e.preventDefault();
        const pos = getPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }

    function stopDraw() {
        drawing = false;
    }

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDraw);
    canvas.addEventListener('mouseleave', stopDraw);
    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDraw);

    clearBtn.addEventListener('click', clearCanvas);

    classifyBtn.addEventListener('click', () => {
        if (!hasDrawn) {
            resultArea.innerHTML = `
                <div class="result-placeholder">
                    <div class="result-atom" style="color: var(--pink)">✗</div>
                    <p style="color: var(--pink)">Please draw a digit first!</p>
                </div>`;
            return;
        }

        resultArea.innerHTML = `
            <div class="processing">
                <p class="processing-text">Running quantum circuit simulation...</p>
                <div class="processing-dots">
                    <span></span><span></span><span></span><span></span>
                </div>
            </div>`;

        const imageDataUrl = canvas.toDataURL('image/png');

        // Call the Flask backend instead of heuristic counting
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageDataUrl })
        })
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            if (data.error) throw new Error(data.error);

            const predicted = data.consensus;
            const algoResults = data.algorithms;

            const gradient = predicted === 2
                ? 'linear-gradient(135deg, #5b7fa6, #8e7ab5)'
                : 'linear-gradient(135deg, #c47a8e, #c9a95a)';

            let barsHtml = '';
            
            // Map the exact algorithm names to frontend representations
            const uiAlgos = {
                'QNN': 'Quantum Neural Network',
                'QCNN': 'Quantum Convolutional Neural Network',
                'VQC': 'Variational Quantum Classifier',
                'VQFE': 'Variational Quantum Feature Embedding',
                'QSVM': 'Quantum Support Vector Machine'
            };

            algoResults.forEach(r => {
                const color = r.confidence > 0.7 ? 'var(--green)' :
                              r.confidence > 0.5 ? 'var(--yellow)' : 'var(--pink)';
                              
                const uiName = uiAlgos[r.name] ? r.name : r.name;
                
                barsHtml += `
                    <div class="result-bar-item">
                        <div class="result-bar-label">
                            <span>${uiName} → Digit ${r.predicted}</span>
                            <span>${(r.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div class="result-bar-track">
                            <div class="result-bar-fill" style="width: 0%; background: ${color};"></div>
                        </div>
                    </div>`;
            });

            resultArea.innerHTML = `
                <div class="classification-result" style="animation: fadeIn 0.4s ease forwards;">
                    <div class="result-header">
                        <div class="result-digit" style="background: ${gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                            ${predicted}
                        </div>
                        <div class="result-label">Consensus Predicted: Digit ${predicted}</div>
                    </div>
                    <div class="result-bars">${barsHtml}</div>
                </div>`;

            requestAnimationFrame(() => {
                setTimeout(() => {
                    const fills = resultArea.querySelectorAll('.result-bar-fill');
                    fills.forEach((fill, i) => {
                        fill.style.width = (algoResults[i].confidence * 100) + '%';
                    });
                }, 50);
            });
        })
        .catch(error => {
            console.error('Error fetching prediction:', error);
            resultArea.innerHTML = `
                <div class="result-placeholder">
                    <div class="result-atom" style="color: var(--pink)">⚠</div>
                    <p style="color: var(--pink)">Backend Error: Make sure app.py is running on port 5000!</p>
                </div>`;
        });
    });
})();


// ============================================================
// GLOSSARY — Expand/Collapse
// ============================================================
(function initGlossary() {
    const items = document.querySelectorAll('.glossary-item');
    items.forEach(item => {
        const term = item.querySelector('.glossary-term');
        term.addEventListener('click', () => {
            const wasOpen = item.classList.contains('open');
            items.forEach(it => it.classList.remove('open'));
            if (!wasOpen) item.classList.add('open');
        });
    });
})();


// ============================================================
// ACCORDION
// ============================================================
(function initAccordion() {
    const items = document.querySelectorAll('.accordion-item');
    items.forEach(item => {
        const header = item.querySelector('.accordion-header');
        header.addEventListener('click', () => {
            const wasOpen = item.classList.contains('open');
            items.forEach(it => it.classList.remove('open'));
            if (!wasOpen) item.classList.add('open');
        });
    });
})();
