import { useEffect, useState } from "react";
import { Activity, ArrowRight, CircleAlert, Database, FileText, Moon, ShieldCheck, SlidersHorizontal, Sun, TrendingUp } from "lucide-react";
const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const DEFAULTS = { pregnancies:1, glucose:120, blood_pressure:70, skin_thickness:20, insulin:80, bmi:25, diabetes_pedigree_function:0.47, age:33 };
const FIELDS = [
  {key:"pregnancies",label:"Pregnancies",min:0,max:20,step:1,unit:"count"},
  {key:"glucose",label:"Glucose",min:40,max:300,step:1,unit:"mg/dL"},
  {key:"blood_pressure",label:"Blood pressure",min:40,max:140,step:1,unit:"mm Hg"},
  {key:"skin_thickness",label:"Skin thickness",min:0,max:100,step:1,unit:"mm"},
  {key:"insulin",label:"Insulin",min:0,max:800,step:1,unit:"µU/mL"},
  {key:"bmi",label:"Body mass index",min:10,max:70,step:0.1,unit:"kg/m²"},
  {key:"diabetes_pedigree_function",label:"Pedigree function",min:0.07,max:2.5,step:0.01,unit:"index"},
  {key:"age",label:"Age",min:18,max:100,step:1,unit:"years"}
];
export default function App(){
  const [theme,setTheme]=useState(()=>localStorage.getItem("pha-theme")||"light");
  const [values,setValues]=useState(DEFAULTS);
  const [api,setApi]=useState("checking");
  const [status,setStatus]=useState(null);
  const [result,setResult]=useState(null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");
  useEffect(()=>{document.documentElement.dataset.theme=theme;localStorage.setItem("pha-theme",theme)},[theme]);
  useEffect(()=>{fetch(API+"/api/status").then(r=>{if(!r.ok)throw new Error();return r.json()}).then(d=>{setStatus(d);setApi("online")}).catch(()=>setApi("offline"))},[]);
  function update(key,value){setValues(old=>({...old,[key]:Number(value)}));setResult(null)}
  async function predict(){
    setBusy(true);setError("");setResult(null);
    try{const response=await fetch(API+"/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(values)});const data=await response.json().catch(()=>({}));if(!response.ok)throw new Error(data.detail||"Prediction request failed.");setResult(data)}
    catch(e){setError(e.message||"Could not connect to the local model API.")}
    finally{setBusy(false)}
  }
  return <div className="health-shell">
    <aside className="sidebar"><a className="brand" href="#home"><span className="brand-icon"><Activity size={22}/></span><span><b>FIELDNOTES</b><small>PUBLIC HEALTH / ML</small></span></a><div className="side-label">WORKSPACE</div><div className="nav-active"><SlidersHorizontal size={17}/> Model exploration <span>01</span></div><div className="side-status"><span className={"dot "+api}/><div><b>{api==="online"?"API connected":api==="checking"?"Checking service":"API unavailable"}</b><small>{status&&status.model_available?"Model: "+status.model_name:"Local trained model required"}</small></div></div><div className="side-bottom"><ShieldCheck size={15}/> Educational research only</div></aside>
    <main className="main" id="home"><header className="topbar"><div><span>DATA SCIENCE</span><i>/</i><strong>Diabetes classification study</strong></div><button className="theme" aria-label="Toggle theme" onClick={()=>setTheme(theme==="light"?"dark":"light")}>{theme==="light"?<Moon size={17}/>:<Sun size={17}/>}</button></header>
      <section className="hero"><div><span className="eyebrow">TABULAR MACHINE LEARNING · PIMA DATASET</span><h1>Explore the<br/><em>model output.</em></h1><p>Adjust input features, submit them to the locally trained scikit-learn pipeline, and review the model's class output with clear limitations.</p></div><div className="hero-badge"><Database size={26}/><b>8</b><span>FEATURES</span></div></section>
      {api==="offline"&&<div className="alert"><CircleAlert size={18}/><span>Python API is offline. Start the local API on port 8000 and refresh.</span></div>}
      {api==="online"&&status&&!status.model_available&&<div className="alert warm"><FileText size={18}/><div><b>No trained model found</b><span>From the repository root, run python src/train.py. This creates a pipeline under models/. Restart the API after training.</span></div></div>}
      {error&&<div className="alert"><CircleAlert size={18}/><span>{error}</span></div>}
      <div className="section-heading"><span>01</span><h2>Input variables</h2><small>EDIT THE SAMPLE VALUES</small></div>
      <section className="workspace"><div className="card input-card"><div className="card-head"><div><small>FEATURE VECTOR</small><h3>Patient-style dataset row</h3></div><button className="reset" onClick={()=>{setValues(DEFAULTS);setResult(null);setError("")}}>Reset values</button></div>
        <div className="fields">{FIELDS.map((f,i)=><label className="field" key={f.key}><span className="field-top"><b>{f.label}</b><small>{f.unit}</small></span><div className="value-row"><input type="range" min={f.min} max={f.max} step={f.step} value={values[f.key]} onChange={e=>update(f.key,e.target.value)}/><input className="number" type="number" min={f.min} max={f.max} step={f.step} value={values[f.key]} onChange={e=>update(f.key,e.target.value)}/></div><span className="range-labels"><span>{f.min}</span><span>{f.max}</span></span></label>)}</div>
        <button className="run" disabled={api!=="online"||!status?.model_available||busy} onClick={predict}>{busy?"Evaluating model…":"Evaluate model output"} <ArrowRight size={16}/></button>
        <p className="fine-print">Input values are sent only to the local API process running on this computer. They are not automatically saved by this interface.</p>
      </div>
      <aside className="right-column"><div className="card protocol"><div className="card-head"><div><small>MODEL CONTEXT</small><h3>What this experiment covers</h3></div><span className="icon-badge"><TrendingUp size={17}/></span></div><div className="protocol-item"><span>01</span><div><b>Preprocessing pipeline</b><small>Median imputation and standard scaling are bundled with the trained model.</small></div></div><div className="protocol-item"><span>02</span><div><b>Model selection</b><small>The saved pipeline is selected from the local models folder after cross-validation training.</small></div></div><div className="protocol-item"><span>03</span><div><b>Interpretation</b><small>Class output reflects the training dataset, not a clinically validated assessment.</small></div></div></div>
        <div className="notice"><ShieldCheck size={18}/><p><b>Important limitation</b><br/>This small, population-specific dataset is not sufficient to estimate an individual's clinical risk. Do not use the output for diagnosis or treatment decisions.</p></div></aside></section>
      <div className="section-heading result-heading"><span>02</span><h2>Model response</h2><small>{result?"LIVE LOCAL INFERENCE":"WAITING FOR INPUT"}</small></div>
      {result?<section className="result-panel"><div className="result-top"><span className="result-icon"><Activity size={21}/></span><div><small>MODEL CLASSIFICATION</small><h3>{result.class_label}</h3></div><span className={"class-tag "+(result.prediction===1?"class-one":"class-zero")}>{result.prediction===1?"CLASS 1":"CLASS 0"}</span></div><div className="probability"><div><span>Class-1 model probability</span><b>{result.class_1_probability==null?"Not available":(result.class_1_probability*100).toFixed(1)+"%"}</b></div>{result.class_1_probability!=null&&<div className="bar"><span style={{width:(result.class_1_probability*100)+"%"}}/></div>}<p>This is the model's output for the selected feature row, not an individual's clinically validated risk.</p></div><div className="result-footer"><span>PIPELINE</span><b>{result.model_name}</b><span>OUTPUT</span><b>{result.class_label}</b></div><div className="notice"><ShieldCheck size={18}/><p>{result.note}</p></div></section>:<div className="empty"><Activity size={26}/><b>No model response yet</b><span>When a trained model is available, its real output will appear here. The interface will not invent metrics or substitute a simulated prediction.</span></div>}
      <footer><span>FIELDNOTES <i>/</i> TABULAR ML</span><span>DATASET-BASED CLASSIFICATION · NOT A CLINICAL TOOL</span></footer>
    </main>
  </div>
}
