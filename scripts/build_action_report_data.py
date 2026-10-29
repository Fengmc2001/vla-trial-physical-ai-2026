#!/usr/bin/env python3
"""Verify saved action handoffs and build monochrome figures and TeX data.

Only saved records are read. No inference, simulator, GPU or robot is started.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from PIL import Image
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from build_academic_report import annotated_gray

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from vla_trial.schema import parse_plan, PlanValidationError

BLACK=colors.black
PLOT_FONT="Times-Roman"
PUBLIC_FILES=("status.json","actions.json","action-milestones.json","evaluation.json","events.jsonl",
              "executed-plan.json","trajectory.json","initial.json","final.json","initial.png","final.png",
              "simulation.mp4","control-events.json","source-manifest.json","runtime-environment.json",
              "inference/model-output.json","inference/plan.json","inference/status.json",
              "inference/events.jsonl","inference/source-manifest.json","inference/runtime-environment.json")
PRIVATE_KEYS={"pid","gpu_uuid","gpu_uuids","host","hostname","operator","allocation_reference",
              "admission_library","model_path","site_config","approval_receipt","admission_output",
              "project_root","cuda_visible_devices"}


def read(p):return json.loads(p.read_text())


def sanitize(value):
    if isinstance(value,dict):return {k:sanitize(v) for k,v in value.items() if k.lower() not in PRIVATE_KEYS}
    if isinstance(value,list):return [sanitize(v) for v in value]
    return value


def export_case(source,dest,files=PUBLIC_FILES):
    for name in files:
        src=source/name
        if not src.is_file():raise ValueError("Missing selected evidence: "+name)
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        if src.suffix==".json":target.write_text(json.dumps(sanitize(read(src)),ensure_ascii=False,indent=2)+"\n")
        elif src.suffix==".jsonl":
            target.write_text("".join(json.dumps(sanitize(json.loads(line)),ensure_ascii=False)+"\n" for line in src.read_text().splitlines() if line.strip()))
        else:target.write_bytes(src.read_bytes())


def verify_case(path,manipulations):
    model=read(path/"inference/model-output.json")
    plan=parse_plan(model["raw_output"],object_ids=["red_block","blue_block","yellow_block"],
                    protected_ids=["red_block"],max_actions=12,allow_repeated_objects=True).to_dict()
    assert plan==read(path/"inference/plan.json")==read(path/"executed-plan.json")["plan"]
    assert hashlib.sha256((path/"initial.png").read_bytes()).hexdigest()==model["image_sha256"]
    assert read(path/"executed-plan.json")["source"]=="QWEN_REAL_INFERENCE"
    status=read(path/"status.json")
    assert status["phase"]=="completed" and status["model_used"] is True and status["video_finalized"] is True
    actions=read(path/"actions.json");milestones=read(path/"action-milestones.json")
    assert len(actions)==len(milestones)==manipulations
    expected=plan["actions"][:-1]
    for a,m,e in zip(actions,milestones,expected):
        assert m["action"]==e and m["completed"] is True
        assert all(a.get(k)==v for k,v in e.items())
    trace=read(path/"trajectory.json")
    assert trace[0]["positions"]==read(path/"initial.json")["object_positions"]
    assert trace[-1]["positions"]==read(path/"final.json")["object_positions"]
    assert all(0 <= trace[i+1]["elapsed_s"]-trace[i]["elapsed_s"] <= .200001 for i in range(len(trace)-1))
    return dict(path=path,model=model,plan=plan,status=status,actions=actions,milestones=milestones,trace=trace)


def axes(d,x,y,w,h,xlim,ylim,xticks,yticks,title):
    def at(t,v):return x+(t-xlim[0])/(xlim[1]-xlim[0])*w,y+(v-ylim[0])/(ylim[1]-ylim[0])*h
    for points in [(x,y,x+w,y),(x,y,x,y+h)]:d.add(Line(*points,strokeWidth=.6,strokeColor=BLACK))
    for v in xticks:
        xx,_=at(v,ylim[0]);d.add(Line(xx,y,xx,y-3,strokeColor=BLACK,strokeWidth=.5))
        d.add(String(xx,y-14,f"{v:g}",fontName=PLOT_FONT,fontSize=9,textAnchor="middle"))
    for v in yticks:
        _,yy=at(xlim[0],v);d.add(Line(x-3,yy,x,yy,strokeColor=BLACK,strokeWidth=.5))
        d.add(String(x-6,yy-3,f"{v:g}",fontName=PLOT_FONT,fontSize=9,textAnchor="end"))
    d.add(String(x,y+h+10,title,fontName=PLOT_FONT,fontSize=10))
    d.add(String(x+w/2,y-29,"Task elapsed time (s)",fontName=PLOT_FONT,fontSize=9,textAnchor="middle"))
    return at


def series(d,rows,object_id,at,dashed=False):
    points=[]
    for row in rows:
        points.extend(at(row["elapsed_s"],row["positions"][object_id][2]*1000))
    d.add(PolyLine(points,strokeColor=BLACK,strokeWidth=.9,strokeDashArray=[5,2] if dashed else None))


def comparison(a,b,dest):
    d=Drawing(475,278)
    for case,y,limit,ticks,title in [(a,178,120,[0,20,40,60,80,100,120],"(a) Case A: block-center height (mm)"),
                                     (b,43,60,[0,10,20,30,40,50,60],"(b) Case B: block-center height (mm), three repetitions")]:
        at=axes(d,40,y,415,79,(0,limit),(0,240),ticks,[0,100,200],title)
        series(d,case["trace"],"blue_block",at)
        if case is a:series(d,case["trace"],"yellow_block",at,True)
        for index,m in enumerate(case["milestones"]):
            xx,yy=at(m["ended_s"],0);_,top=at(m["ended_s"],240)
            d.add(Line(xx,yy,xx,top,strokeColor=BLACK,strokeWidth=.35,strokeDashArray=[1,3]))
    renderPDF.drawToFile(d,str(dest))


def pipeline(dest):
    d=Drawing(475,136)
    rows=[[("Actual image + instruction",0),("Qwen inference",164),("strict JSON -> TaskPlan",328)],
          [("live scene state -> IK",328),("whitelisted dispatch",164),("plan.json + exit code",0)]]
    for row,y in zip(rows,[91,31]):
        for text,x in row:
            d.add(Rect(x,y,146,30,fillColor=colors.white,strokeColor=BLACK,strokeWidth=.65))
            d.add(String(x+73,y+11,text,fontName=PLOT_FONT,fontSize=9,textAnchor="middle"))
    def arrow(x,y,xx,yy):
        d.add(Line(x,y,xx,yy,strokeColor=BLACK,strokeWidth=.65))
        theta=math.atan2(yy-y,xx-x)
        for t in [-.45,.45]:d.add(Line(xx,yy,xx-4*math.cos(theta+t),yy-4*math.sin(theta+t),strokeColor=BLACK,strokeWidth=.65))
    arrow(146,106,164,106);arrow(310,106,328,106)
    d.add(Line(401,91,401,77,strokeColor=BLACK,strokeWidth=.6))
    d.add(Line(401,77,73,77,strokeColor=BLACK,strokeWidth=.6));arrow(73,77,73,61)
    arrow(146,46,164,46);arrow(310,46,328,46)
    d.add(String(237,12,"P1: model process  ->  local file IPC  ->  P0: control process + Genesis",fontName=PLOT_FONT,fontSize=9,textAnchor="middle"))
    renderPDF.drawToFile(d,str(dest))


def frame(case,t,dest):
    raw=dest.with_name(dest.stem+"-source.png")
    offset=read(case["path"]/"initial.json")["simulation_seconds"]-.1
    subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-ss",f"{t+offset:.4f}","-i",str(case["path"]/"simulation.mp4"),
                    "-frames:v","1","-threads","1",str(raw)],check=True,capture_output=True,timeout=30)
    sample=min(case["trace"],key=lambda r:abs(r["elapsed_s"]-t))
    annotated_gray(raw,dest,sample["positions"],label_offsets={"blue_block":(-110,-50),"yellow_block":(65,-12),"red_block":(-125,60)})
    with Image.open(dest) as im:im.crop((300,210,730,660)).save(dest)
    return {"image":dest.name,"task_elapsed_s":t,"raw_video_pts_s":t+offset,"sample_elapsed_s":sample["elapsed_s"],
            "crop_box":[300,210,730,660],"mode":"L","sha256":hashlib.sha256(dest.read_bytes()).hexdigest()}


def main():
    global PLOT_FONT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run",type=Path,default=ROOT/"examples/validated_multistep_run")
    parser.add_argument("--new-run",type=Path,default=ROOT/"outputs/repeat-blue-user-20261029")
    parser.add_argument("--rejected-run",type=Path,default=ROOT/"outputs/repeat-blue-rejected-20261029")
    parser.add_argument("--output",type=Path,default=ROOT/"交付/action通信実験レポート")
    parser.add_argument("--plot-font",type=Path)
    args=parser.parse_args();out=args.output;images=out/"images";images.mkdir(parents=True,exist_ok=True)
    plot_font=args.plot_font or Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
    if plot_font.is_file():
        pdfmetrics.registerFont(TTFont("ActionPlotSerif",str(plot_font)))
        PLOT_FONT="ActionPlotSerif"
    a=verify_case(args.old_run,5);b=verify_case(args.new_run,3)
    assert [x['action'] for x in b['plan']['actions']]==['pick_place']*3+['done']
    assert all(x['lift_verified'] and x['placement_verified'] for x in b['actions'])
    assert read(args.new_run/"evaluation.json")["evaluation_scope"]=="custom_task_unscored"
    failed=read(args.rejected_run/"inference/model-output.json")
    try:parse_plan(failed['raw_output'],object_ids=['red_block','blue_block','yellow_block'],protected_ids=['red_block'],max_actions=12,allow_repeated_objects=True)
    except PlanValidationError as error:rejected_error=str(error)
    else:raise AssertionError("The recorded incomplete plan must be rejected")
    assert rejected_error=="Plan must terminate with done."
    assert not any(json.loads(row).get('phase')=='moving' for row in (args.rejected_run/'events.jsonl').read_text().splitlines() if row.strip())
    am=read(args.old_run/'source-manifest.json');bm=read(args.new_run/'source-manifest.json')
    modules=['qwen.py','schema.py','execute.py','simulation.py']
    assert all(am['src/vla_trial/'+name]==bm['src/vla_trial/'+name] for name in modules)
    changes=[k for k in am if am[k]!=bm.get(k)]
    pipeline(images/'pipeline.pdf');comparison(a,b,images/'comparison.pdf')
    image_sources=[frame(a,80.2,images/'a-stack.png'),frame(a,120,images/'a-final.png')]
    peaks=[]
    for index,m in enumerate(b['milestones']):
        rows=[r for r in b['trace'] if m['started_s']-1e-8<=r['elapsed_s']<=m['ended_s']+1e-8]
        peak=max(rows,key=lambda r:r['positions']['blue_block'][2]);peaks.append(peak)
        image_sources.append(frame(b,peak['elapsed_s'],images/f'b-lift-{index+1}.png'))
    (out/'figure-sources.json').write_text(json.dumps(image_sources,ensure_ascii=False,indent=2)+'\n')
    # Exact saved UTF-8 bytes are reported as file sizes, never as network throughput.
    macros=[]
    def macro(name,value):macros.append('\\newcommand{\\'+name+'}{'+str(value)+'}')
    for prefix,case in [('A',a),('B',b)]:
        macro('Case'+prefix+'RawBytes',len(case['model']['raw_output'].encode('utf-8')))
        macro('Case'+prefix+'PlanBytes',(case['path']/'inference/plan.json').stat().st_size)
        macro('Case'+prefix+'DispatchBytes',(case['path']/'executed-plan.json').stat().st_size)
        macro('Case'+prefix+'Samples',len(case['trace']))
        macro('Case'+prefix+'Duration',f"{case['trace'][-1]['elapsed_s']:.1f}")
    def vec(p):return '('+', '.join(f'{x*1000:.3f}' for x in p)+')'
    btable=[]
    for i,(action,m,peak) in enumerate(zip(b['actions'],b['milestones'],peaks)):
        btable.append(f"{i+1} & {m['started_s']:.1f}--{m['ended_s']:.1f} & {vec(action['position_before'])} & {vec(action['position_after'])} & {peak['positions']['blue_block'][2]*1000:.3f} \\\\")
        macro('PeakTime'+['One','Two','Three'][i],f"{peak['elapsed_s']:.1f}")
    macro('RepeatStateRows','\n'.join(btable))
    boundary=[]
    for index in [0,1]:
        previous=b['actions'][index]['position_after'];current=b['actions'][index+1]['position_before']
        assert previous==current
        boundary.append(math.dist(previous,current)*1000)
    macro('RepeatHandoffMax',f'{max(boundary):.3f}')
    macro('InitialVsSecondDistance',f"{math.dist(b['actions'][0]['position_before'],b['actions'][1]['position_before'])*1000:.3f}")
    st=a['actions'][3]['positions_after'];nxt=a['actions'][4]['positions_before'];assert st==nxt
    macro('StackBlue',vec(st['blue_block']));macro('StackYellow',vec(st['yellow_block']))
    macro('FinalBlue',vec(a['actions'][4]['positions_after']['blue_block']))
    macro('FinalYellow',vec(a['actions'][4]['positions_after']['yellow_block']))
    macro('StackGap',f"{(st['yellow_block'][2]-st['blue_block'][2])*1000:.3f}")
    macro('FinalGap',f"{(a['actions'][4]['positions_after']['yellow_block'][2]-a['actions'][4]['positions_after']['blue_block'][2])*1000:.3f}")
    red0=b['trace'][0]['positions']['red_block']
    red=max(math.dist(r['positions']['red_block'],red0) for r in b['trace'])
    macro('RepeatRedDisplacement',f'{red:.3e}')
    (out/'data.tex').write_text('\n'.join(macros)+'\n')
    selected_b=ROOT/'examples/additional_repeat_run';export_case(args.new_run,selected_b)
    selected_c=ROOT/'examples/rejected_plan_example'
    export_case(args.rejected_run,selected_c,('status.json','events.jsonl','source-manifest.json','inference/model-output.json','inference/status.json','inference/events.jsonl'))
    (selected_b/'README.md').write_text('# Additional user-generated repetition case\n\nThe user entered 抓起蓝色方块然后放下。重复三回. Actual Qwen inference selected three blue-to-A pick_place actions. All three manipulations completed; lift and placement checks passed. This is a custom task with evaluation_scope=custom_task_unscored and success=null, not a second pass of the fixed five-action course task. No original-position return was requested explicitly or verified. The controller/model/parser source hashes match the original run; admission.py differs.\n')
    (selected_c/'README.md').write_text('# Recorded rejected plan\n\nFor the user instruction 抓起蓝色方块，重复三回, the model returned three actions without the required final done. The strict parser rejected the plan. No manipulation phase was entered. This is an observed parsing failure, not a completed robot task.\n')
    evidence={'case_a_actions':5,'case_b_actions':3,'case_b_completed':True,'case_b_semantic_success':None,
              'case_b_evaluation_scope':'custom_task_unscored','case_c_reproduced_rejection':rejected_error,
              'case_c_no_manipulation_phase':True,'core_modules_hashes_match':modules,'changed_manifest_files':changes,
              'repeat_handoff_distance_mm':boundary,'stack_to_transport_positions_exactly_equal':True,
              'new_red_maximum_displacement_m':red,'figures':image_sources}
    report_dir=ROOT/'reports/action-report-review';report_dir.mkdir(parents=True,exist_ok=True)
    (report_dir/'evidence-check.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in evidence.items() if k!='figures'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
