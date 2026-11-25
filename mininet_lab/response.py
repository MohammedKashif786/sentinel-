"""Response actions applied to the Open vSwitch switches (used when you run the demo inside Mininet).
   Monitor -> nothing changes;  Restrict -> host may only talk to the server;  Isolate -> all its traffic is dropped."""
import subprocess

def _ovs(sw, *args): subprocess.run(["ovs-ofctl", *args], check=False)

def apply(sw, host_ip, server_ip, level):
    clear(sw, host_ip)
    if level == 2:      # RESTRICT
        _ovs(sw, "add-flow", sw, f"priority=100,ip,nw_src={host_ip},nw_dst={server_ip},actions=normal")
        _ovs(sw, "add-flow", sw, f"priority=90,ip,nw_src={host_ip},actions=drop")
    elif level == 3:    # ISOLATE
        _ovs(sw, "add-flow", sw, f"priority=100,ip,nw_src={host_ip},actions=drop")
        _ovs(sw, "add-flow", sw, f"priority=100,ip,nw_dst={host_ip},actions=drop")

def clear(sw, host_ip):   # RESTORE access
    _ovs(sw, "del-flows", sw, f"ip,nw_src={host_ip}"); _ovs(sw, "del-flows", sw, f"ip,nw_dst={host_ip}")
