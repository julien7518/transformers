'use client';
import Link from 'next/link';
import {useEffect, useRef, useState} from 'react';
import {characterCount, isTranslation, validateInput, type Direction, type Health, type Translation} from '@/lib/contracts';

const examples = {
  to_genz: ['Ce film est vraiment excellent.', 'Je suis vraiment déçu.', 'Je suis très motivé pour ce projet.'],
  to_standard: ['Ce film est une dinguerie.', 'J’ai trop le seum.', 'Je suis grave chaud pour ce projet.'],
};
function Arrow({swap = false}: {swap?: boolean}) {return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">{swap ? <><path d="M4 8h16m-4-4 4 4-4 4M20 16H4m4-4-4 4 4 4"/></> : <path d="M4 12h16m-6-6 6 6-6 6"/>}</svg>;}
export default function Translator({mock}: {mock: boolean}) {
  const [text,setText] = useState('');
  const [direction,setDirection] = useState<Direction>('to_genz');
  const [result,setResult] = useState<Translation | null>(null);
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState('');
  const [notice,setNotice] = useState('');
  const [health,setHealth] = useState<Health | null>(null);
  const [healthError,setHealthError] = useState(false);
  const generation = useRef(0);
  const active = useRef<AbortController | null>(null);
  const source = useRef<HTMLTextAreaElement>(null);
  const copyTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const limit = health?.limits.max_input_chars ?? 500;
  const count = characterCount(text);
  async function refreshHealth() {
    try { const response = await fetch('/api/health', {cache:'no-store'}); const data = await response.json();
      if (!response.ok || !['ready','unavailable'].includes(data.status)) throw new Error();
      setHealth(data); setHealthError(false);
    } catch {setHealth(null); setHealthError(true);}
  }
  useEffect(() => {const timer = setTimeout(() => {void refreshHealth();}, 0); return () => {clearTimeout(timer); active.current?.abort(); if(copyTimer.current) clearTimeout(copyTimer.current);};}, []);
  function invalidate() {generation.current++; active.current?.abort(); active.current = null; setBusy(false); setResult(null); setError(''); setNotice('');}
  function changeText(value: string) {invalidate(); setText(value);}
  function changeDirection(value: Direction) {if(value !== direction) {invalidate(); setDirection(value);}}
  function swap() {const nextText = result?.translation ?? text; invalidate(); setDirection(direction === 'to_genz' ? 'to_standard' : 'to_genz'); setText(nextText); source.current?.focus();}
  async function translate(event: React.FormEvent) {
    event.preventDefault();
    if (busy) return;
    const invalid = validateInput(text,limit);
    if(invalid) {setError(invalid); source.current?.focus(); return;}
    const id = ++generation.current;
    const controller = new AbortController(); active.current = controller;
    setBusy(true); setError(''); setNotice(''); setResult(null);
    try {
      const response = await fetch('/api/translate', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text.trim(),direction}),signal:controller.signal});
      const data = await response.json();
      if(id !== generation.current) return;
      if(!response.ok) throw new Error(data.error?.message || 'La traduction a échoué. Réessaie dans un instant.');
      if(!isTranslation(data) || data.direction !== direction) throw new Error('La réponse du modèle est invalide. Réessaie dans un instant.');
      setResult(data);
    } catch (failure) {if(id === generation.current && !controller.signal.aborted) setError(failure instanceof Error ? failure.message : 'Une erreur est survenue. Réessaie.');}
    finally {if(id === generation.current) {setBusy(false); active.current = null;}}
  }
  async function copy() {if(!result) return; try {await navigator.clipboard.writeText(result.translation); setNotice('Traduction copiée.'); if(copyTimer.current) clearTimeout(copyTimer.current); copyTimer.current = setTimeout(()=>setNotice(''),3000);} catch {setNotice('Copie impossible. Sélectionne le texte pour le copier.');}}
  const ready = health?.status === 'ready';
  return <div className="shell">
    <header className="header"><Link className="brand" href="/" aria-label="Entre nous, accueil"><span className="brand-icon">en<span>↗</span></span><span>entre nous<span className="brand-dot">.</span></span></Link><span className="project-tag">UN PROJET ÉTUDIANT · UN VRAI TRANSFORMER</span><a className="about-link" href="#apropos">Le projet <span aria-hidden="true">↗</span></a></header>
    <main>
      <section className="intro" aria-labelledby="title"><div className="eyebrow"><span/> LES MÊMES IDÉES, UN AUTRE REGISTRE</div><h1 id="title">Le français change<br/>de <span className="tone">ton<span className="underline"/></span><span className="period">.</span></h1><p>De « c’est excellent » à « c’est une dinguerie ».<br className="desktop-break"/> Passe du français standard au langage Gen Z, et inversement.</p></section>
      {mock && <div className="mock-banner" role="status">Mode démonstration — les exemples affichent des réponses simulées, pas les sorties du modèle.</div>}
      <section className="workspace" aria-label="Traducteur">
        <div className="toolbar"><div className="direction-options" role="group" aria-label="Sens de traduction"><button type="button" aria-pressed={direction==='to_genz'} onClick={()=>changeDirection('to_genz')}>Standard <span aria-hidden="true">→</span> Gen Z</button><button type="button" aria-pressed={direction==='to_standard'} onClick={()=>changeDirection('to_standard')}>Gen Z <span aria-hidden="true">→</span> Standard</button></div><button className="swap" type="button" onClick={swap} aria-label="Inverser le sens de traduction" title="Inverser le sens"><Arrow swap/></button></div>
        <form onSubmit={translate} noValidate>
          <div className="panels"><div className="panel source-panel"><div className="panel-heading"><label htmlFor="source">{direction==='to_genz' ? 'Français standard' : 'Français Gen Z'}</label><span className="panel-kicker">TON TEXTE</span></div><textarea ref={source} id="source" value={text} onChange={e=>changeText(e.target.value)} placeholder={direction==='to_genz' ? 'Alors, qu’est-ce qu’on raconte ?' : 'Balance ta phrase ici…'} aria-describedby="character-count input-hint" aria-invalid={count>limit || (!!error && !text.trim())} spellCheck/><div className="input-footer"><button type="button" className="text-button" onClick={()=>{changeText('');source.current?.focus();}} disabled={!text}>Effacer</button><span id="character-count" className={count>limit?'over-limit':''}>{count} <span>/ {limit}</span></span></div></div>
          <div className={`panel result-panel ${result?'has-result':''}`} aria-busy={busy}><div className="panel-heading"><h2>{direction==='to_genz'?'Français Gen Z':'Français standard'}</h2><span className="panel-kicker">{result?'TRADUCTION':'LE RÉSULTAT'}</span></div>{busy ? <div className="empty-result"><span className="loader"/><p>On change de registre…</p><span>Le modèle prépare sa réponse.</span></div> : result ? <><p className="translation">{result.translation || 'Le modèle n’a produit aucun texte.'}</p><div className="result-footer"><span>{result.truncated?'Génération limitée':'À toi de voir si le ton est juste.'}</span><button type="button" className="copy-button" onClick={copy}>Copier <span aria-hidden="true">⧉</span></button></div></> : <div className="empty-result"><span className="result-symbol" aria-hidden="true">a<span>↗</span></span><p>Un autre ton, le même message.</p><span>Ta traduction apparaîtra ici.</span></div>}</div></div>
          <div className="action-bar"><p id="input-hint"><span aria-hidden="true">✳</span> Une phrase courte, c’est un bon début.</p><button type="submit" className="translate-button" disabled={busy}>{busy?'Traduction en cours':'Changer de ton'}{busy?<span className="button-loader"/>:<Arrow/>}</button></div>
        </form>
        <div className="live-messages" aria-live="polite" aria-atomic="true">{error && <p className="error-message">{error}</p>}{notice && <p className="notice-message">{notice}</p>}{result?.truncated && <p className="warning-message">Le modèle a atteint sa limite de génération. La phrase peut être incomplète.</p>}{busy && <span className="sr-only">Traduction en cours.</span>}</div>
      </section>
      <section className="examples" aria-label="Exemples à essayer"><p>En panne d’inspiration ?</p><div>{examples[direction].map(example=><button type="button" key={example} onClick={()=>{changeText(example);source.current?.focus();}}>{example}<span aria-hidden="true">↗</span></button>)}</div></section>
      <section className="project-info" id="apropos"><div><span className="small-label">SOUS LE CAPOT</span><h2>Petit modèle.<br/>Grand terrain d’apprentissage.</h2></div><div className="project-description"><p>Un Transformer encodeur-décodeur construit et entraîné depuis zéro avec PyTorch. Deux registres, un seul modèle, et aucune génération externe.</p><p className="disclaimer">C’est un laboratoire, pas un dictionnaire : les résultats sont expérimentaux et peuvent manquer de justesse. Le sens passe avant l’argot.</p></div></section>
    </main>
    <footer><span>entre nous. <span className="footer-caption">Le langage est un terrain de jeu.</span></span><div className="model-status"><span className={`status-dot ${ready?'ready':''}`}/><span>{mock?'Démonstration simulée':ready?'Modèle connecté':healthError?'API injoignable':health?.status==='unavailable'?'Modèle non chargé':'Connexion au modèle…'}</span>{(!ready || healthError) && <button type="button" className="text-button" onClick={()=>void refreshHealth()}>Réessayer</button>}{ready && !mock && <span className="model-id">{health.model_id}</span>}</div></footer>
  </div>;
}
