import { useCallback, useEffect, useState } from "react";
import Papa from "papaparse";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const tabs = [
  { id: "run", label: "Nouvelle analyse" },
  { id: "report", label: "Résultats" },
  { id: "selection", label: "Sélection" },
  { id: "benchmark", label: "Méthode" },
] as const;
type Tab = (typeof tabs)[number]["id"];

type Run = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  agent: string;
  m: number;
  error?: string | null;
};
type TestRecord = { p_value: number; id: string; test: string; variables: string[] };
type Claim = {
  claim: string;
  variables?: string[];
  asserts_effect?: boolean;
  status?: string;
  reason?: string | null;
  evidence?: TestRecord[];
  false?: boolean;
  supported_by_engine?: boolean;
  dataset_seed?: number;
};
type Report = {
  m: number;
  R_brut: number;
  confirmed_count: number;
  claims: Claim[];
  rejected: { claim: string; reason: string }[];
  selection_diagnostic?: {
    full_registry_ks_uniform?: { statistic: number; p_value: number; n: number } | null;
    registry_vs_reported_ks?: {
      statistic: number;
      p_value: number;
      n_registry: number;
      n_reported: number;
    } | null;
  };
};
type Registry = { tests: TestRecord[] };
type Summary = {
  mode: string;
  seeds: number;
  n: number;
  summary: Record<
    string,
    {
      mean?: number;
      min?: number;
      max?: number;
      datasets_with_any?: number;
      reported?: number;
      false?: number;
      fdr_pooled?: number;
      recall_mean?: number;
    }
  >;
};
type Demo = {
  provenance: string;
  datasets: number;
  archive_summary: {
    claims: number;
    effect_claims: number;
    false_effect_claims: number;
    retained_by_legacy_engine: number;
  };
  claims: Claim[];
  null_benchmark: Summary;
  planted_benchmark: Summary;
};
type E4Benchmark = {
  datasets: number;
  n: number;
  budgets: Record<
    string,
    {
      mean_false_claims: number;
      median_false_claims: number;
      bootstrap_ci: [number, number];
    }
  >;
  result_paragraph: string;
};
type E6Benchmark = {
  datasets: number;
  n: number;
  methods: Record<
    string,
    { recall_mean: number; fdp: number; discoveries: number }
  >;
  result_paragraph: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Erreur API (${response.status})`);
  }
  return response.json() as Promise<T>;
}

async function optionalRequest<T>(path: string): Promise<T | null> {
  const response = await fetch(`${API_URL}${path}`);
  if (response.status === 404) return null;
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Erreur API (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function parseCsv(text: string): Record<string, string | number | boolean | null>[] {
  const parsed = Papa.parse<Record<string, string | number | boolean | null>>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: "greedy",
  });
  if (parsed.errors.length) throw new Error(`CSV invalide : ${parsed.errors[0].message}`);
  if (!parsed.meta.fields?.length || parsed.meta.fields.some((name) => !name.trim())) {
    throw new Error("Le CSV doit contenir des noms de colonnes valides.");
  }
  if (!parsed.data.length) throw new Error("Le CSV doit contenir au moins une ligne.");
  return parsed.data.map((row) =>
    Object.fromEntries(
      Object.entries(row).map(([key, value]) => [key, value === "" ? null : value]),
    ),
  );
}

function App() {
  const [active, setActive] = useState<Tab>("run");
  const [demo, setDemo] = useState<Demo | null>(null);
  const [e4, setE4] = useState<E4Benchmark | null>(null);
  const [e6, setE6] = useState<E6Benchmark | null>(null);
  const [loadingBenchmarks, setLoadingBenchmarks] = useState(false);
  const [demoMode, setDemoMode] = useState(true);
  const [run, setRun] = useState<Run | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [registryTests, setRegistryTests] = useState<TestRecord[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [maxTests, setMaxTests] = useState(25);
  const [prompt, setPrompt] = useState("");
  const [agent, setAgent] = useState<"deterministic" | "muse" | "anthropic">("deterministic");
  const [demoSeed, setDemoSeed] = useState<number | "all">("all");
  const [error, setError] = useState("");
  const [loadingDemo, setLoadingDemo] = useState(true);

  const loadDemo = useCallback(async () => {
    setLoadingDemo(true);
    try {
      setDemo(await request<Demo>("/demo"));
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Impossible de charger la démo.");
    } finally {
      setLoadingDemo(false);
    }
  }, []);

  useEffect(() => {
    void loadDemo();
  }, [loadDemo]);

  useEffect(() => {
    if (active !== "benchmark") return;
    let activeEffect = true;
    setLoadingBenchmarks(true);
    Promise.all([
      optionalRequest<E4Benchmark>("/benchmark/E4"),
      optionalRequest<E6Benchmark>("/benchmark/E6"),
    ])
      .then(([nextE4, nextE6]) => {
        if (!activeEffect) return;
        setE4(nextE4);
        setE6(nextE6);
      })
      .catch((reason: Error) => {
        if (activeEffect) setError(reason.message);
      })
      .finally(() => {
        if (activeEffect) setLoadingBenchmarks(false);
      });
    return () => {
      activeEffect = false;
    };
  }, [active]);

  useEffect(() => {
    if (!run || run.status === "completed" || run.status === "failed") return;
    const source = new EventSource(`${API_URL}/runs/${run.id}/events`);
    source.onmessage = (event) => {
      const update = JSON.parse(event.data) as Run;
      setRun(update);
      if (update.status === "completed") {
        source.close();
        setActive("report");
        void Promise.all([
          request<Report>(`/runs/${update.id}/report`),
          request<Registry>(`/runs/${update.id}/registry`),
        ]).then(([nextReport, registry]) => {
          setReport(nextReport);
          setRegistryTests(registry.tests);
        }).catch((reason: Error) => setError(reason.message));
      }
      if (update.status === "failed") {
        source.close();
        setError(update.error ?? "Le run a échoué.");
      }
    };
    source.onerror = () => {
      source.close();
      setError("Flux du run interrompu. Vérifiez que l’API est disponible.");
    };
    return () => source.close();
  }, [run]);

  const currentReport = report;
  async function startRun() {
    if (!file) {
      setError("Choisissez un fichier CSV pour lancer une analyse.");
      return;
    }
    try {
      const dataset = parseCsv(await file.text());
      setError("");
      setReport(null);
      setRegistryTests([]);
      setDemoMode(false);
      const created = await request<Run>("/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset,
          agent,
          config: { prompt, max_tests: maxTests },
        }),
      });
      setRun(created);
      setActive("run");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Échec de création du run.");
    }
  }

  async function replayDemo() {
    try {
      setError("");
      setReport(null);
      setRegistryTests([]);
      setDemoMode(false);
      const created = await request<Run>("/demo/run", { method: "POST" });
      setRun(created);
      setActive("run");
    } catch (reason) {
      setDemoMode(true);
      setError(reason instanceof Error ? reason.message : "Impossible de lancer la démo moteur.");
    }
  }

  const activeRun = !demoMode ? run : null;
  const archivedClaims = demo?.claims.filter((claim) =>
    demoSeed === "all" || claim.dataset_seed === demoSeed,
  ) ?? [];
  const displayedClaims = demoMode ? archivedClaims : currentReport?.claims ?? [];
  const m = activeRun?.m ?? 0;
  const falsePositiveRisk = demoMode ? null : 1 - Math.pow(0.95, m);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" onClick={() => setActive("run")}>
          <span className="brand-mark">Da</span>
          <span>data alchemist<small>STATISTICAL AUDIT</small></span>
        </a>
      </aside>

      <main className="main">
        <header className="topbar">
          <a className="brand top-brand" href="#" onClick={() => setActive("run")}>
            <span className="brand-mark">Da</span>
            <span>data alchemist<small>STATISTICAL AUDIT</small></span>
          </a>
          <div className="topbar-right">
            <span className="privacy-label">◈ Vos lignes restent locales</span>
            {active !== "run" && <button className="new-analysis" onClick={() => { setActive("run"); setRun(null); setReport(null); setDemoMode(false); }}>+ Nouvelle analyse</button>}
          </div>
        </header>

        {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError("")}>Fermer</button></div>}

        {active !== "run" && <section className="page-heading">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" />STATISTICAL DISCOVERY AUDIT</div>
            <h1>{heading(active)}</h1>
            <p>{subheading(active)}</p>
          </div>
        </section>}

        {active === "run" && (
          <RunView
            demoMode={demoMode}
            demo={demo}
            loadingDemo={loadingDemo}
            agent={agent}
            run={activeRun}
            m={m}
            risk={falsePositiveRisk === null ? "n/a" : formatPercent(falsePositiveRisk)}
            file={file}
            maxTests={maxTests}
            prompt={prompt}
            onFile={(nextFile) => { setFile(nextFile); if (nextFile) setDemoMode(false); }}
            onMaxTests={setMaxTests}
            onPrompt={setPrompt}
            onAgent={setAgent}
            onReplayDemo={replayDemo}
            onStart={startRun}
            onShowArchive={() => { setDemoMode(true); setActive("report"); }}
            onShowResults={() => setActive("report")}
          />
        )}
        {active === "report" && <ReportView claims={displayedClaims} demoMode={demoMode} report={currentReport} provenance={demo?.provenance} archiveSummary={demo?.archive_summary} demoSeed={demoSeed} onDemoSeed={setDemoSeed} demoDatasets={demo?.datasets ?? 0} />}
        {active === "selection" && <SelectionView report={demoMode ? null : currentReport} records={registryTests} demoMode={demoMode} />}
        {active === "benchmark" && <BenchmarkView demo={demo} e4={e4} e6={e6} loading={loadingBenchmarks} />}
        <footer className="footer"><span>DATA ALCHEMIST <b>·</b> AUDIT STATISTIQUE</span><nav aria-label="Informations"><button onClick={() => setActive("selection")}>Diagnostic de sélection</button><button onClick={() => setActive("benchmark")}>Méthode & benchmarks</button></nav><span>Association ≠ causalité.</span></footer>
      </main>
    </div>
  );
}

function heading(tab: Tab) {
  return { run: "Déposez. Analysez. Décidez.", report: "Voici ce que les données soutiennent.", selection: "Diagnostic de sélection.", benchmark: "Mesurer avant d’affirmer." }[tab];
}
function subheading(tab: Tab) {
  return {
    run: "Un audit statistique de vos données, pas un tableau de bord de plus.",
    report: "Le moteur confronte les affirmations de l’agent aux tests réellement enregistrés.",
    selection: "Comparez la distribution des tests lancés à celle des résultats mis en avant.",
    benchmark: "Résultats archivés uniquement — aucun chiffre de benchmark n’est simulé.",
  }[tab];
}
function formatPercent(value: number) { return `${(value * 100).toFixed(1)}%`; }

function RunView(props: {
  demoMode: boolean; demo: Demo | null; loadingDemo: boolean;
  run: Run | null; m: number; risk: string;
  file: File | null; maxTests: number; prompt: string;
  onReplayDemo: () => void;
  agent: "deterministic" | "muse" | "anthropic";
  onAgent: (value: "deterministic" | "muse" | "anthropic") => void;
  onFile: (file: File | null) => void; onMaxTests: (value: number) => void;
  onPrompt: (value: string) => void; onStart: () => void;
  onShowArchive: () => void;
  onShowResults: () => void;
}) {
  const { demoMode, demo, loadingDemo, run, m, risk, file, maxTests, prompt, onFile, onMaxTests, onPrompt, onStart, onReplayDemo, agent, onAgent, onShowArchive, onShowResults } = props;
  const done = run?.status === "completed";
  return (
    <>
      <section className="product-intro">
        <div className="eyebrow"><span className="eyebrow-line" />DATA ALCHEMIST · ANALYSE PRIVÉE</div>
        <h1>Vos données, <em>passées au crible.</em></h1>
        <p>Importez un CSV. L’agent explore, le moteur vérifie chaque affirmation et vous livre un rapport traçable.</p>
        <div className="steps"><span className="step-active"><b>01</b> Données</span><i /><span className={run ? "step-active" : ""}><b>02</b> Exploration</span><i /><span className={done ? "step-active" : ""}><b>03</b> Résultats</span></div>
      </section>
      <div className="run-grid">
        <article className="panel upload-panel">
          <div className="panel-kicker">ÉTAPE 1 · IMPORTER</div>
          <h2>Commencez avec un jeu de données</h2>
          <p className="muted">Fichier CSV avec une ligne d’en-tête. Vos lignes restent sur votre instance.</p>
          <label className="dropzone">
            <input type="file" accept=".csv,text/csv" onChange={(event) => onFile(event.target.files?.[0] ?? null)} />
            <span className="upload-icon">↥</span>
            <strong>{file?.name ?? "Déposez un CSV ou parcourez vos fichiers"}</strong>
            <small>{file ? `${(file.size / 1024).toFixed(1)} Ko` : "CSV · Taille raisonnable recommandée"}</small>
          </label>
          <label className="field-label" htmlFor="agent-select">Analyste</label>
          <select id="agent-select" value={agent} onChange={(event) => onAgent(event.target.value as typeof agent)}>
            <option value="deterministic">Analyste déterministe (hors ligne)</option>
            <option value="muse">Muse Spark (clé Meta requise)</option>
            <option value="anthropic">Anthropic (clé API requise)</option>
          </select>
          <label className="field-label" htmlFor="test-budget">Budget exploratoire <span>{maxTests}</span></label>
          <input id="test-budget" className="range-input" type="range" min="1" max="200" value={maxTests} onChange={(event) => onMaxTests(Number(event.target.value))} />
          <small className="muted">Les tests de confirmation s’ajoutent au registre m et sont comptés en plus.</small>
          <label className="field-label" htmlFor="analysis-prompt">Consigne (facultative)</label>
          <textarea id="analysis-prompt" value={prompt} onChange={(event) => onPrompt(event.target.value)} placeholder="Ex. soyez prudent face aux comparaisons multiples…" rows={2} />
          <button className="primary-button" disabled={demoMode || run?.status === "running"} onClick={onStart}>
            {run?.status === "running" ? <><span className="spinner" /> Analyse en cours…</> : "Analyser mes données"} <span>↗</span>
          </button>
          {demoMode && <div className="inline-note">Choisissez votre CSV pour démarrer une analyse privée.</div>}
        </article>
        <article className="sample-card">
          <div className="sample-orbit"><span>Da</span><i>✳</i><b>p</b></div>
          <div className="panel-kicker">PAS DE FICHIER SOUS LA MAIN ?</div>
          <h2>Voyez l’audit en action.</h2>
          <p>Rejouez une analyse reproductible hors ligne, sans compte ni clé API.</p>
          <button className="secondary-button" onClick={onReplayDemo}>Lancer la démo guidée <span>↗</span></button>
          {demo && <button className="archive-link" onClick={onShowArchive}>Consulter les résultats historiques ({demo.archive_summary.claims} affirmations)</button>}
          {loadingDemo && <small className="loading-note">Préparation de la démo…</small>}
          {!demoMode && run && <div className="run-progress"><span className={run.status === "running" ? "working" : ""} />{run.status === "completed" ? `${m} tests audités · rapport prêt` : run.status === "failed" ? "Analyse en échec" : `Analyse ${run.status} · ${m} tests enregistrés`}{run.status === "completed" && <strong>{risk}*</strong>}</div>}
          {done && <button className="result-link" onClick={onShowResults}>Lire le rapport d’audit →</button>}
        </article>
      </div>
      <div className="trust-line"><span>◇</span> Chaque test est enregistré <i /> Les résultats sont corrigés pour la multiplicité <i /> Aucune association n’est présentée comme causale</div>
    </>
  );
}

function Metric({ label, value, note, accent = false }: { label: string; value: string; note: string; accent?: boolean }) {
  return <article className={`metric-card ${accent ? "metric-accent" : ""}`}><div>{label}</div><strong>{value}</strong><small>{note}</small></article>;
}

function ReportView({ claims, demoMode, report, provenance, archiveSummary, demoSeed, onDemoSeed, demoDatasets }: {
  claims: Claim[]; demoMode: boolean; report: Report | null; provenance?: string;
  archiveSummary?: Demo["archive_summary"]; demoSeed: number | "all";
  onDemoSeed: (value: number | "all") => void; demoDatasets: number;
}) {
  const supported = claims.filter((claim) => claim.status === "supported").length;
  const rejected = claims.filter((claim) => claim.status === "rejected" || (demoMode && claim.asserts_effect)).length;
  return (
    <>
      {demoMode && <div className="provenance"><strong>Archive historique</strong><span>{provenance ?? "Ces résultats proviennent d’un benchmark antérieur au registre instrumenté."}</span></div>}
      {demoMode && archiveSummary && <div className="metrics-grid compact"><Metric label="Affirmations archivées" value={String(archiveSummary.claims)} note={`${demoDatasets} jeux synthétiques nuls`} /><Metric label="Fausses affirmations d’effet" value={String(archiveSummary.false_effect_claims)} note="Étiquetées par la vérité terrain historique" accent /><Metric label="Retenues par l’ancien moteur" value={String(archiveSummary.retained_by_legacy_engine)} note="Champ d’archive, non recalculé aujourd’hui" /></div>}
      <div className="compare-grid">
        <article className="compare-card agent-card"><div className="panel-kicker">SORTIE DE L’AGENT</div><strong>{claims.length}</strong><span>affirmations archivées</span><div className="compare-foot">Aucune affirmation n’est considérée vraie par défaut.</div></article>
        <div className="compare-arrow">→</div>
        <article className="compare-card engine-card"><div className="panel-kicker">RETENU PAR LE MOTEUR</div><strong>{demoMode ? "0" : String(supported)}</strong><span>affirmations étayées</span><div className="compare-foot">{demoMode ? "Le registre historique n’est pas disponible pour recalculer ces preuves." : `${rejected} rejetée(s) · ${report?.confirmed_count ?? 0} confirmée(s)`}</div></article>
      </div>
      {!demoMode && report && <div className="metrics-grid compact"><Metric label="m · tests lancés" value={String(report.m)} note="Entrées du registre" /><Metric label="R brut · p &lt; 0,05" value={String(report.R_brut)} note="Avant correction de multiplicité" /><Metric label="Confirmées" value={String(report.confirmed_count)} note="Sur données de confirmation" /></div>}
      <article className="panel table-panel">
        <div className="panel-topline">
          <div><div className="panel-kicker">DÉCISION PAR AFFIRMATION</div><h2>{claims.length} affirmations examinées</h2></div>
          {demoMode ? <label className="table-filter">Jeu<select value={demoSeed} onChange={(event) => onDemoSeed(event.target.value === "all" ? "all" : Number(event.target.value))}><option value="all">Tous</option>{Array.from({ length: demoDatasets }, (_, index) => <option value={index} key={index}>Jeu {index + 1}</option>)}</select></label> : <span className="table-count">REGISTRE ACTIF</span>}
        </div>
        <div className="table-scroll"><table><thead><tr><th>Affirmation</th><th>Variables</th><th>Résultat</th><th>Justification</th></tr></thead><tbody>
          {claims.map((claim, index) => {
            const status = demoMode ? (claim.asserts_effect && claim.false ? "false" : "untested") : claim.status ?? "untested";
            return <tr key={`${claim.dataset_seed}-${claim.claim}-${index}`}><td className="claim-cell">{claim.claim}</td><td><div className="tag-list">{(claim.variables ?? []).map((variable) => <span className="variable-tag" key={variable}>{variable}</span>)}</div></td><td><StatusBadge status={status} /></td><td className="reason-cell">{demoMode ? (claim.asserts_effect ? `Jeu ${Number(claim.dataset_seed ?? 0) + 1}: affirmation d’effet fausse selon la vérité terrain; p-values exploratoires non archivées.` : `Jeu ${Number(claim.dataset_seed ?? 0) + 1}: archive historique sans registre complet pour auditer cette affirmation.`) : claim.reason ?? (claim.evidence?.length ? `${claim.evidence.length} test(s) référencé(s)` : "Aucun test enregistré.")}</td></tr>;
          })}
          {claims.length === 0 && <tr><td colSpan={4} className="empty-row">Aucune affirmation disponible. Lancez une analyse ou chargez la démo.</td></tr>}
        </tbody></table></div>
      </article>
    </>
  );
}

function StatusBadge({ status }: { status: string }) {
  const normalized = status === "supported" ? "supported" : status === "rejected" || status === "false" ? "rejected" : "untested";
  const text = normalized === "supported" ? "ÉTAYÉE" : status === "false" ? "FAUSSE" : normalized === "rejected" ? "REJETÉE" : "NON TESTÉE";
  return <span className={`claim-badge ${normalized}`}><i />{text}</span>;
}

function SelectionView({ report, records, demoMode }: { report: Report | null; records: TestRecord[]; demoMode: boolean }) {
  if (demoMode) return <div className="empty-state"><div className="empty-icon">⌁</div><h2>Pas de registre historique à comparer</h2><p>Les résultats archivés contiennent des affirmations, mais pas les p-values de tous les tests exploratoires. Les inventer rendrait ce diagnostic trompeur.</p><span>Désactivez le mode démo et lancez une analyse instrumentée.</span></div>;
  const registryP = records.map((record) => record.p_value).filter((value) => Number.isFinite(value) && value >= 0 && value <= 1);
  const reportedP = (report?.claims ?? []).filter((claim) => claim.status === "supported").flatMap((claim) => claim.evidence ?? []).map((record) => record.p_value);
  const bins = Array.from({ length: 10 }, (_, index) => {
    const start = index / 10;
    return {
      bin: `${start.toFixed(1)}–${((index + 1) / 10).toFixed(1)}`,
      registry: registryP.filter((value) => value >= start && (index === 9 ? value <= 1 : value < start + 0.1)).length,
      reported: reportedP.filter((value) => value >= start && (index === 9 ? value <= 1 : value < start + 0.1)).length,
    };
  });
  const diagnostic = report?.selection_diagnostic?.registry_vs_reported_ks;
  return (
    <>
      <div className="metrics-grid"><Metric label="P-values enregistrées" value={String(registryP.length)} note="Issues des tests du registre" /><Metric label="P-values des affirmations étayées" value={String(reportedP.length)} note="Sous-ensemble référencé par l’agent" /><Metric label="KS · registre vs rapportées" value={diagnostic ? diagnostic.p_value.toFixed(3) : "n/a"} note={diagnostic ? `p = ${diagnostic.p_value.toFixed(3)} · descriptif, non causal` : "Échantillon insuffisant pour le test KS"} /></div>
      <article className="panel chart-panel"><div className="panel-topline"><div><div className="panel-kicker">DISTRIBUTION DES P-VALUES</div><h2>Tous les tests vs résultats rapportés</h2></div><span className="table-count">ANALYSE DESCRIPTIVE</span></div>
        <div className="large-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={bins} barGap={2}><CartesianGrid stroke="#e9eeea" vertical={false} /><XAxis dataKey="bin" tickLine={false} axisLine={false} /><YAxis allowDecimals={false} tickLine={false} axisLine={false} /><Tooltip /><Legend /><Bar dataKey="registry" name="Registre complet" fill="#287767" radius={[4, 4, 0, 0]} /><Bar dataKey="reported" name="Affirmations étayées" fill="#e1a85d" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div>
        <div className="ks-note"><span className="ks-symbol">KS</span><p>{diagnostic ? <>Test de Kolmogorov–Smirnov à deux échantillons : <strong>D = {diagnostic.statistic.toFixed(3)}, p = {diagnostic.p_value.toFixed(3)}</strong>. La non-significativité ne prouve pas l’absence de sélection.</> : "Le diagnostic KS n’est pas disponible pour cet échantillon."} La p-curve est descriptive; elle ne corrige pas les biais de sélection.</p></div>
      </article>
    </>
  );
}

function BenchmarkView({
  demo,
  e4,
  e6,
  loading,
}: {
  demo: Demo | null;
  e4: E4Benchmark | null;
  e6: E6Benchmark | null;
  loading: boolean;
}) {
  if (!demo) return <div className="empty-state"><h2>Chargement des archives</h2><p>Le serveur API fournit les résultats enregistrés, sans requête à un modèle externe.</p></div>;
  const nullRows = Object.entries(demo.null_benchmark.summary).map(([method, values]) => ({ method: shortMethod(method), falsePerDataset: values.mean ?? 0 }));
  const plantedRows = Object.entries(demo.planted_benchmark.summary).map(([method, values]) => ({ method: shortMethod(method), falseDiscoveryRate: values.fdr_pooled ?? 0, recall: values.recall_mean ?? 0 }));
  const e4Rows = e4
    ? Object.entries(e4.budgets)
        .map(([budget, values]) => ({
          budget: Number(budget),
          mean: values.mean_false_claims,
          bootstrapError: [
            Math.max(0, values.mean_false_claims - values.bootstrap_ci[0]),
            Math.max(0, values.bootstrap_ci[1] - values.mean_false_claims),
          ],
        }))
        .sort((left, right) => left.budget - right.budget)
    : [];
  const e6Rows = e6
    ? Object.entries(e6.methods).map(([method, values]) => ({
        method: shortMethod(method),
        fdp: values.fdp,
        recall: values.recall_mean,
      }))
    : [];
  return (
    <>
      <div className="benchmark-grid">
        <article className="panel chart-panel">
          <div className="panel-kicker">E4 · DOSE-RÉPONSE</div>
          <h2>Budget et fausses affirmations</h2>
          {e4 ? (
            <>
              <p className="muted">{e4.result_paragraph}</p>
              <div className="chart-box">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={e4Rows}>
                    <CartesianGrid stroke="#e9eeea" vertical={false} />
                    <XAxis dataKey="budget" name="Budget de tests" tickLine={false} axisLine={false} />
                    <YAxis allowDecimals={false} name="Affirmations fausses" tickLine={false} axisLine={false} />
                    <Tooltip />
                    <Line type="monotone" dataKey="mean" name="Moyenne par jeu" stroke="#287767" strokeWidth={2}>
                      <ErrorBar dataKey="bootstrapError" width={4} strokeWidth={1} />
                    </Line>
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <small className="muted">Barres : IC bootstrap à 95 %, rééchantillonnage par jeu.</small>
            </>
          ) : (
            <Pending label="E4 non disponible" detail={loading ? "Chargement des résultats versionnés…" : "Aucun results/e4.json. Exécutez le protocole après validation du gate E0."} />
          )}
        </article>
        <article className="panel chart-panel">
          <div className="panel-kicker">E6 · MÉTHODES SUR FAMILLE FIXE</div>
          <h2>Rappel et fausses découvertes</h2>
          {e6 ? (
            <>
              <p className="muted">{e6.result_paragraph}</p>
              <div className="chart-box">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={e6Rows}>
                    <CartesianGrid stroke="#e9eeea" vertical={false} />
                    <XAxis dataKey="method" tickLine={false} axisLine={false} />
                    <YAxis domain={[0, 1]} tickLine={false} axisLine={false} />
                    <Tooltip formatter={(value: number) => formatPercent(value)} />
                    <Legend />
                    <Bar dataKey="fdp" name="FDP agrégée" fill="#d18470" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="recall" name="Rappel moyen" fill="#287767" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <small className="muted">Les garanties sont conditionnelles aux hypothèses indiquées dans le JSON du benchmark.</small>
            </>
          ) : (
            <Pending label="E6 non disponible" detail={loading ? "Chargement des résultats versionnés…" : "Aucun results/e6.json. Exécutez le protocole après validation du gate E0."} />
          )}
        </article>
      </div>
      <div className="archive-note"><span>ARCHIVE · {demo.null_benchmark.seeds} GRAINES</span><p>Résultats issus du benchmark historique existant ({demo.null_benchmark.n.toLocaleString("fr-FR")} lignes par jeu). Ils précèdent le protocole E0–E6 et ne doivent pas être interprétés comme sa validation.</p></div>
      <div className="benchmark-grid">
        <article className="panel chart-panel"><div className="panel-kicker">JEUX NULS</div><h2>Découvertes par jeu</h2><p className="muted">Moyenne des découvertes rapportées; {demo.null_benchmark.seeds} jeux seulement.</p><div className="chart-box"><ResponsiveContainer width="100%" height="100%"><BarChart data={nullRows} layout="vertical" margin={{ left: 12, right: 16 }}><CartesianGrid stroke="#e9eeea" horizontal={false} /><XAxis type="number" tickLine={false} axisLine={false} /><YAxis type="category" dataKey="method" width={155} tickLine={false} axisLine={false} /><Tooltip /><Bar dataKey="falsePerDataset" name="Moyenne par jeu" fill="#287767" radius={[0, 5, 5, 0]}>{nullRows.map((_, index) => <Cell key={index} fill={index === 0 ? "#cb796a" : "#287767"} />)}</Bar></BarChart></ResponsiveContainer></div></article>
        <article className="panel chart-panel"><div className="panel-kicker">RELATIONS PLANTÉES</div><h2>FDP et rappel observés</h2><p className="muted">FDP agrégée sur les découvertes et rappel moyen, selon la méthode.</p><div className="chart-box"><ResponsiveContainer width="100%" height="100%"><BarChart data={plantedRows}><CartesianGrid stroke="#e9eeea" vertical={false} /><XAxis dataKey="method" tickLine={false} axisLine={false} /><YAxis domain={[0, 1]} tickLine={false} axisLine={false} /><Tooltip formatter={(value: number) => formatPercent(value)} /><Legend /><Bar dataKey="falseDiscoveryRate" name="FDP agrégée" fill="#d18470" radius={[4, 4, 0, 0]} /><Bar dataKey="recall" name="Rappel moyen" fill="#287767" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div></article>
      </div>
    </>
  );
}

function shortMethod(method: string) {
  if (method.startsWith("naive")) return "Brut · p < 0,05";
  if (method.includes("FDR control")) return "BH";
  if (method.includes("replication")) return "Réplication";
  return "Filtre confondeurs";
}
function Pending({ label, detail }: { label: string; detail: string }) {
  return <article className="pending-card"><span>EN ATTENTE</span><strong>{label}</strong><p>{detail}</p></article>;
}

export default App;
