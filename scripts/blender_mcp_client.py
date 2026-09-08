"""Project-local client for the installed Blender Lab MCP add-on's actual TCP bridge.

Protocol verified from mcp_to_blender_server.py v1.0.0: UTF-8 JSON + NUL,
type=execute, code, strict_json boolean. This is the add-on transport, not a
claim that a separate Codex MCP tool was registered. Keeps request/response evidence.
"""
import argparse, json, socket, datetime, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def execute(code, label='request', timeout=1800):
    request={'type':'execute','strict_json':True,'code':code}
    port=int(os.environ.get('VP_MCP_PORT','9876'))
    with socket.create_connection(('127.0.0.1',port),timeout=10) as conn:
        conn.settimeout(timeout);conn.sendall(json.dumps(request).encode('utf-8')+b'\0')
        buf=bytearray()
        while b'\0' not in buf:
            chunk=conn.recv(65536)
            if not chunk:raise RuntimeError('Blender MCP bridge closed without a complete response')
            buf.extend(chunk)
    response=json.loads(buf.split(b'\0',1)[0])
    record={'time':datetime.datetime.now().isoformat(),'transport':'Blender Lab MCP 1.0.0 TCP bridge','request':request,'response':response}
    (ROOT/f'logs/mcp_{label}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    if response.get('status')!='ok':raise RuntimeError(json.dumps(response,indent=2))
    return response

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--script');p.add_argument('--test',action='store_true');p.add_argument('--inspect',action='store_true');args=p.parse_args()
    if args.inspect:
        code="import bpy\nresult={'version':bpy.app.version_string,'file':bpy.data.filepath,'objects':len(bpy.data.objects),'camera':bpy.context.scene.camera.name,'engine':bpy.context.scene.render.engine,'mcp_addons':[x for x in bpy.context.preferences.addons.keys() if 'mcp' in x]}"
        label='inspect'
    else:
        path=(ROOT/args.script).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('Script must be inside project')
        argv=[str(path)]+(['--test'] if args.test else [])
        code=f"import sys\nfrom pathlib import Path\nsys.argv={argv!r}\nscript=Path({str(path)!r})\nnamespace={{'__file__':str(script),'__name__':'__main__'}}\nexec(compile(script.read_text(encoding='utf-8'),str(script),'exec'),namespace)\nresult={{'script':str(script),'complete':True}}"
        label=path.stem+('_test' if args.test else '_final')
    response=execute(code,label)
    print(json.dumps(response,indent=2))
