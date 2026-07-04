import React, { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  BarChart3,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Database,
  ExternalLink,
  Github,
  Layers,
  Menu,
  Search,
  ShieldCheck,
  X,
  Zap
} from "lucide-react";

const repoUrl = "https://github.com/Technet43/MASS-AI-Project";

const navItems = [
  { href: "#platform", label: "Platform" },
  { href: "#workflow", label: "Workflow" },
  { href: "#screens", label: "Screens" },
  { href: "#deploy", label: "Deploy" },
  { href: repoUrl, label: "GitHub", external: true }
];

const metrics = [
  { label: "Customers scored", value: "2,000", tone: "blue" },
  { label: "Stacking ROC-AUC", value: "0.9428", tone: "green" },
  { label: "Theft patterns", value: "8", tone: "red" },
  { label: "Case workflow", value: "Ops", tone: "amber" }
];

const capabilities = [
  {
    icon: ShieldCheck,
    title: "Risk scoring",
    text: "Ranks suspicious smart meter accounts with model scores, risk bands, and case-ready signals."
  },
  {
    icon: BarChart3,
    title: "Decision dashboard",
    text: "Surfaces risk distribution, watchlists, model quality, and operational exposure for quick triage."
  },
  {
    icon: Search,
    title: "Investigation view",
    text: "Connects each account to evidence, feature profiles, notes, and recommended next action."
  },
  {
    icon: Database,
    title: "Synthetic demo engine",
    text: "Generates credible pilot data when live utility data is restricted or not ready to share."
  },
  {
    icon: Layers,
    title: "Multi-model path",
    text: "Combines Isolation Forest, tree ensembles, gradient boosting, and stacking for stronger screening."
  },
  {
    icon: Zap,
    title: "Realtime direction",
    text: "Keeps the product story ready for live telemetry, gateway ingestion, and faster feature processing."
  }
];

const workflow = [
  {
    step: "01",
    title: "Ingest",
    text: "Load scored customer records, synthetic scenarios, or future long-format smart meter telemetry."
  },
  {
    step: "02",
    title: "Score",
    text: "Generate model probabilities, anomaly features, risk bands, and theft-pattern hints."
  },
  {
    step: "03",
    title: "Prioritize",
    text: "Move the highest-risk accounts into a watchlist that analysts can review without model noise."
  },
  {
    step: "04",
    title: "Act",
    text: "Create investigation records, capture notes, export evidence, and move cases through an ops flow."
  }
];

const gallery = [
  {
    title: "Operations overview",
    image: "/images/dashboard_overview_current.png",
    text: "KPI cards, risk bands, trend context, and work queue indicators for the web dashboard."
  },
  {
    title: "Risk alerts",
    image: "/images/overview_risk_alerts_current.png",
    text: "A focused view of high-priority accounts and the active risk mix across the demo workspace."
  },
  {
    title: "Time-series comparison",
    image: "/images/comparison_timeseries_current.png",
    text: "Consumption patterns side by side, making abnormal behavior easier to explain."
  },
  {
    title: "Statistical comparison",
    image: "/images/comparison_statistics_current.png",
    text: "Feature-level comparison for normal and suspicious accounts."
  },
  {
    title: "Customer detail",
    image: "/images/customer_detail_current.png",
    text: "Investigation-ready customer page with risk context and analyst evidence."
  },
  {
    title: "Model quality",
    image: "/images/performance_curves_current.png",
    text: "Performance curves for presenting model behavior in a pilot review."
  }
];

const signalCards = [
  { label: "Critical risk", value: "143", change: "+18 today", tone: "red" },
  { label: "Field priority", value: "37", change: "ready", tone: "amber" },
  { label: "Model confidence", value: "91%", change: "stacked", tone: "green" }
];

const riskFeed = [
  { id: "TR-0917", region: "Metro", score: "94", status: "tamper pattern" },
  { id: "TR-2284", region: "Coastal", score: "88", status: "peak clipping" },
  { id: "TR-4062", region: "Plateau", score: "82", status: "bypass signal" }
];

const techStack = [
  "React",
  "Vite",
  "Vercel",
  "Python",
  "Streamlit",
  "scikit-learn",
  "XGBoost",
  "SQLite",
  "Ops Center"
];

function IconButton({ children, href, variant = "primary", external = false }) {
  return (
    <a
      className={`button button-${variant}`}
      href={href}
      target={external ? "_blank" : undefined}
      rel={external ? "noreferrer" : undefined}
    >
      {children}
    </a>
  );
}

function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeImage, setActiveImage] = useState(null);
  const dialogRef = useRef(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (activeImage !== null && !dialog.open) {
      dialog.showModal();
    }

    if (activeImage === null && dialog.open) {
      dialog.close();
    }
  }, [activeImage]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (activeImage === null) return;
      if (event.key === "ArrowLeft") {
        setActiveImage((activeImage - 1 + gallery.length) % gallery.length);
      }
      if (event.key === "ArrowRight") {
        setActiveImage((activeImage + 1) % gallery.length);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeImage]);

  const activeSlide = activeImage === null ? null : gallery[activeImage];

  return (
    <div className="site-shell">
      <header className="site-header">
        <a className="brand" href="#top" onClick={() => setMenuOpen(false)}>
          <span className="brand-symbol" aria-hidden="true">
            M
          </span>
          <span>
            MASS-AI
            <small>Smart meter intelligence</small>
          </span>
        </a>

        <button
          className="nav-toggle"
          type="button"
          aria-label="Toggle navigation"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((open) => !open)}
        >
          {menuOpen ? <X size={22} /> : <Menu size={22} />}
        </button>

        <nav className={`site-nav ${menuOpen ? "open" : ""}`} aria-label="Primary navigation">
          {navItems.map((item) => (
            <a
              key={item.label}
              href={item.href}
              target={item.external ? "_blank" : undefined}
              rel={item.external ? "noreferrer" : undefined}
              onClick={() => setMenuOpen(false)}
            >
              {item.label}
              {item.external ? <ExternalLink size={15} aria-hidden="true" /> : null}
            </a>
          ))}
        </nav>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">Pilot intelligence platform</p>
            <h1>MASS-AI detects energy theft signals before they become losses.</h1>
            <p className="hero-text">
              Smart meter anomaly scoring, analyst prioritization, and case workflow presented as a
              serious utility product. The React site is ready for Vercel; the prototype stays connected
              through the real dashboard assets already in the repo.
            </p>
            <div className="hero-actions">
              <IconButton href="#screens">
                Explore console
                <ArrowRight size={18} aria-hidden="true" />
              </IconButton>
              <IconButton href={repoUrl} variant="secondary" external>
                <Github size={18} aria-hidden="true" />
                Repository
              </IconButton>
            </div>
            <div className="hero-proof" aria-label="MASS-AI headline proof points">
              <span>multi-model scoring</span>
              <span>ops-ready cases</span>
              <span>Vercel deploy</span>
            </div>
          </div>

          <div className="command-center" aria-label="MASS-AI product preview">
            <div className="command-topbar">
              <span>MASS-AI Command Center</span>
              <strong>Live pilot view</strong>
            </div>

            <div className="command-grid">
              <div className="console-panel hero-product">
                <div className="product-toolbar">
                  <span>Risk workspace</span>
                  <strong>active</strong>
                </div>
                <img
                  src="/images/dashboard_overview_current.png"
                  alt="MASS-AI dashboard overview with risk metrics and operational panels"
                />
              </div>

              <div className="console-panel risk-console">
                <div className="panel-heading">
                  <span>Risk feed</span>
                  <strong>now</strong>
                </div>
                {riskFeed.map((item) => (
                  <div className="risk-row" key={item.id}>
                    <div>
                      <strong>{item.id}</strong>
                      <span>{item.region} / {item.status}</span>
                    </div>
                    <b>{item.score}</b>
                  </div>
                ))}
              </div>

              <div className="console-panel signal-panel">
                {signalCards.map((item) => (
                  <div className={`signal-card signal-${item.tone}`} key={item.label}>
                    <span>{item.label}</span>
                    <strong>{item.value}</strong>
                    <small>{item.change}</small>
                  </div>
                ))}
              </div>

              <div className="scan-strip" aria-hidden="true">
                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        </section>

        <section className="metric-band" aria-label="Project metrics">
          {metrics.map((metric) => (
            <div className="metric" key={metric.label}>
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
            </div>
          ))}
        </section>

        <section className="section split-section" id="platform">
          <div className="section-copy">
            <p className="eyebrow">Platform</p>
            <h2>The product story now feels like software a utility team could open.</h2>
            <p>
              MASS-AI packages model scoring, explainable signals, and case workflow into a product
              story that can be shown to energy teams without exposing restricted production data.
            </p>
          </div>

          <div className="capability-grid">
            {capabilities.map((item) => {
              const Icon = item.icon;
              return (
                <article className="capability-card" key={item.title}>
                  <Icon size={24} aria-hidden="true" />
                  <h3>{item.title}</h3>
                  <p>{item.text}</p>
                </article>
              );
            })}
          </div>
        </section>

        <section className="section workflow-section" id="workflow">
          <div className="section-copy">
            <p className="eyebrow">Workflow</p>
            <h2>From telemetry to field-ready action.</h2>
          </div>

          <div className="workflow-grid">
            {workflow.map((item) => (
              <article className="workflow-card" key={item.step}>
                <span>{item.step}</span>
                <h3>{item.title}</h3>
                <p>{item.text}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="section screens-section" id="screens">
          <div className="section-copy">
            <p className="eyebrow">Screens</p>
            <h2>Real dashboard screenshots, framed like a product launch.</h2>
            <p>
              The gallery is built from the screenshots already committed in the repository, so the
              Vercel site stays aligned with the current MASS-AI prototype.
            </p>
          </div>

          <div className="gallery-grid">
            {gallery.map((item, index) => (
              <button
                className="gallery-item"
                type="button"
                key={item.title}
                onClick={() => setActiveImage(index)}
              >
                <img src={item.image} alt={`${item.title} screenshot`} />
                <span>{item.title}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="section deploy-section" id="deploy">
          <div className="deploy-copy">
            <p className="eyebrow">Vercel</p>
            <h2>Ready for import and deploy.</h2>
            <p>
              The web directory contains the Vite React app, Vercel configuration, and static
              assets. Vercel can publish it with the default Node build flow.
            </p>
          </div>

          <div className="deploy-panel">
            <div className="deploy-row">
              <CheckCircle2 size={20} aria-hidden="true" />
              <span>Framework preset</span>
              <strong>Vite</strong>
            </div>
            <div className="deploy-row">
              <CheckCircle2 size={20} aria-hidden="true" />
              <span>Build command</span>
              <strong>npm run build</strong>
            </div>
            <div className="deploy-row">
              <CheckCircle2 size={20} aria-hidden="true" />
              <span>Output directory</span>
              <strong>dist</strong>
            </div>
          </div>
        </section>

        <section className="section stack-section">
          <div className="section-copy">
            <p className="eyebrow">Stack</p>
            <h2>Frontend and prototype stack.</h2>
          </div>
          <div className="stack-list">
            {techStack.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        </section>

        <section className="closing-section">
          <div>
            <p className="eyebrow">Next step</p>
            <h2>Import the repository into Vercel when GitHub is ready.</h2>
          </div>
          <IconButton href={repoUrl} variant="secondary" external>
            <Github size={18} aria-hidden="true" />
            Open GitHub
          </IconButton>
        </section>
      </main>

      <footer className="site-footer">
        <span>MASS-AI by Omer Burak Kocak</span>
        <span>React website prepared for Vercel deployment</span>
      </footer>

      <dialog
        className="lightbox"
        ref={dialogRef}
        onClose={() => setActiveImage(null)}
        onClick={(event) => {
          if (event.target === event.currentTarget) setActiveImage(null);
        }}
      >
        {activeSlide ? (
          <>
            <div className="lightbox-header">
              <button
                type="button"
                aria-label="Previous image"
                onClick={() => setActiveImage((activeImage - 1 + gallery.length) % gallery.length)}
              >
                <ChevronLeft size={22} aria-hidden="true" />
              </button>
              <span>
                {activeImage + 1} / {gallery.length}
              </span>
              <button
                type="button"
                aria-label="Next image"
                onClick={() => setActiveImage((activeImage + 1) % gallery.length)}
              >
                <ChevronRight size={22} aria-hidden="true" />
              </button>
              <button type="button" aria-label="Close preview" onClick={() => setActiveImage(null)}>
                <X size={22} aria-hidden="true" />
              </button>
            </div>
            <img src={activeSlide.image} alt={`${activeSlide.title} enlarged screenshot`} />
            <div className="lightbox-copy">
              <h3>{activeSlide.title}</h3>
              <p>{activeSlide.text}</p>
            </div>
          </>
        ) : null}
      </dialog>
    </div>
  );
}

export default App;
