<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { getContext } from "svelte";
    import { getGraphData, searchGraph } from "$lib/apis/graph";

    const i18n = getContext("i18n");

    let svgEl: SVGElement;
    let container: HTMLDivElement;
    let stats = { nodes: 0, relationships: 0 };
    let selectedNode: any = null;
    let searchQuery = "";
    let loading = true;
    let error = "";
    let simulation: any = null;
    let allNodes: any[] = [];
    let allEdges: any[] = [];
    let searchMode = false;
    let activeFilter = "All";

    // ── Mini-mode (floating panel over chat) ─────────────────────────────────
    let miniMode = true;   // start in mini mode
    let miniW = 520;
    let miniH = 420;
    let miniX = window.innerWidth  - miniW - 20;
    let miniY = window.innerHeight - miniH - 60;
    let dragging = false;
    let dragOffX = 0;
    let dragOffY = 0;

    function startDrag(e: MouseEvent) {
        dragging = true;
        dragOffX = e.clientX - miniX;
        dragOffY = e.clientY - miniY;
        window.addEventListener("mousemove", onDrag);
        window.addEventListener("mouseup",   stopDrag);
    }
    function onDrag(e: MouseEvent) {
        if (!dragging) return;
        miniX = Math.max(0, Math.min(window.innerWidth  - miniW, e.clientX - dragOffX));
        miniY = Math.max(0, Math.min(window.innerHeight - miniH, e.clientY - dragOffY));
    }
    function stopDrag() {
        dragging = false;
        window.removeEventListener("mousemove", onDrag);
        window.removeEventListener("mouseup",   stopDrag);
    }
    function toggleMode() {
        miniMode = !miniMode;
        setTimeout(() => renderGraph(allNodes, allEdges), 100);
    }

    const NODE_COLORS: Record<string, string> = {
        Country: "#6366f1", Region: "#8b5cf6", River: "#06b6d4",
        Mountain: "#10b981", Place: "#f59e0b", Organization: "#06b6d4",
        Concept: "#f59e0b", Location: "#ef4444", Climate: "#ec4899",
        Crop: "#84cc16", Industry: "#a78bfa", Resource: "#fb923c",
        Event: "#34d399", Person: "#10b981", Entity: "#94a3b8",
    };
    function getColor(t: string) { return NODE_COLORS[t] || NODE_COLORS.Entity; }

    function resetSearch() {
        searchMode = false; searchQuery = ""; error = "";
        renderGraph(allNodes, allEdges);
    }

    async function loadGraph() {
        loading = true; error = "";
        try {
            const token = localStorage.getItem("token") || "";
            const data = await getGraphData(token);
            if (data.error) { error = data.error; return; }
            allNodes = data.nodes || [];
            allEdges = data.edges || [];
            stats    = data.stats || { nodes: 0, relationships: 0 };
            renderGraph(allNodes, allEdges);
        } catch (e: any) { error = e.message; }
        finally { loading = false; }
    }

    async function handleSearch() {
        if (!searchQuery.trim()) { renderGraph(allNodes, allEdges); return; }
        loading = true; searchMode = true;
        try {
            const token = localStorage.getItem("token") || "";
            const data  = await searchGraph(token, searchQuery);
            if (data.nodes?.length > 0) renderGraph(data.nodes, data.edges || []);
            else { renderGraph([], []); error = "No entities found for: " + searchQuery; }
        } finally { loading = false; }
    }

    function filterByType(type: string) {
        activeFilter = type;
        if (type === "All") { renderGraph(allNodes, allEdges); return; }
        const filtered = allNodes.filter(n => n.type === type);
        const ids  = new Set(filtered.map((n: any) => n.id));
        const edgs = allEdges.filter((e: any) => ids.has(e.source) || ids.has(e.target));
        renderGraph(filtered, edgs);
    }

    function renderGraph(nodes: any[], edges: any[]) {
        if (!svgEl || !window.d3) return;
        const d3 = window.d3;
        const W  = container.clientWidth;
        const H  = container.clientHeight;

        d3.select(svgEl).selectAll("*").remove();
        if (simulation) simulation.stop();

        const svg = d3.select(svgEl).attr("width", W).attr("height", H);
        const g   = svg.append("g");

        svg.call(d3.zoom().scaleExtent([0.1, 6]).on("zoom", (e: any) => {
            g.attr("transform", e.transform);
        }));

        // Deduplicate nodes by id before rendering
        const uniqueNodeMap = new Map<string, any>();
        for (const n of nodes) {
            if (!uniqueNodeMap.has(n.id)) {
                uniqueNodeMap.set(n.id, { ...n, x: W/2 + Math.random()*200-100, y: H/2 + Math.random()*200-100 });
            }
        }
        const nodeMap  = uniqueNodeMap;
        const nodeList = Array.from(nodeMap.values());

        const validEdges = edges
            .filter((e: any) => nodeMap.has(String(e.source)) && nodeMap.has(String(e.target)))
            .map((e: any) => ({ ...e, source: nodeMap.get(String(e.source)), target: nodeMap.get(String(e.target)) }));

        svg.append("defs").append("marker")
            .attr("id","arrow").attr("viewBox","0 -5 10 10")
            .attr("refX",20).attr("refY",0)
            .attr("markerWidth",6).attr("markerHeight",6).attr("orient","auto")
            .append("path").attr("d","M0,-5L10,0L0,5").attr("fill","#475569");

        const link = g.append("g").selectAll("line").data(validEdges).enter().append("line")
            .attr("stroke","#475569").attr("stroke-opacity",0.4)
            .attr("stroke-width",1).attr("marker-end","url(#arrow)");

        const linkLabel = g.append("g").selectAll("text").data(validEdges).enter().append("text")
            .attr("font-size","7px").attr("fill","#64748b").attr("text-anchor","middle")
            .text((d: any) => (d.type||"")
                .replace("HAS_CONCEPT","concept").replace("RELATED_TO","related")
                .replace("CO_OCCURS","co-occurs").replace("CONTAINS","in")
                .replace("ARE_COMMON_TREES","trees").replace("IS_PARTICIPATED_BY","part of")
                .replace("AT_THE_SAME_TIME","same time").replace("CONSISTS_OF","consists"));

        const node = g.append("g").selectAll("circle").data(nodeList).enter().append("circle")
            .attr("r", (d: any) => d.highlighted ? 12 : 7)
            .attr("fill", (d: any) => getColor(d.type))
            .attr("stroke", (d: any) => d.highlighted ? "#fff" : "rgba(255,255,255,0.15)")
            .attr("stroke-width", (d: any) => d.highlighted ? 2 : 0.5)
            .style("cursor","pointer")
            .on("click", (_: any, d: any) => { selectedNode = d; })
            .call(d3.drag()
                .on("start", (e: any, d: any) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
                .on("drag",  (e: any, d: any) => { d.fx=e.x; d.fy=e.y; })
                .on("end",   (e: any, d: any) => { if (!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; })
            );

        const label = g.append("g").selectAll("text").data(nodeList).enter().append("text")
            .attr("font-size","9px").attr("fill","#e2e8f0")
            .attr("text-anchor","middle").attr("dy","18px")
            .text((d: any) => { const l = d.label||d.id||""; return l.length>22 ? l.slice(0,22)+"…" : l; });

        simulation = d3.forceSimulation(nodeList)
            .force("link",      d3.forceLink(validEdges).distance(80).strength(0.4))
            .force("charge",    d3.forceManyBody().strength(-160))
            .force("center",    d3.forceCenter(W/2, H/2))
            .force("collision", d3.forceCollide(22))
            .on("tick", () => {
                link.attr("x1",(d:any)=>d.source.x).attr("y1",(d:any)=>d.source.y)
                    .attr("x2",(d:any)=>d.target.x).attr("y2",(d:any)=>d.target.y);
                linkLabel.attr("x",(d:any)=>(d.source.x+d.target.x)/2)
                         .attr("y",(d:any)=>(d.source.y+d.target.y)/2);
                node.attr("cx",(d:any)=>d.x).attr("cy",(d:any)=>d.y);
                label.attr("x",(d:any)=>d.x).attr("y",(d:any)=>d.y);
            });
    }

    onMount(async () => {
        if (!window.d3) {
            await new Promise<void>(resolve => {
                const s = document.createElement("script");
                s.src = "https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js";
                s.onload = () => resolve();
                document.head.appendChild(s);
            });
        }
        await loadGraph();
        const urlSearch = new URLSearchParams(window.location.search).get("search");
        if (urlSearch) { searchQuery = urlSearch; await handleSearch(); }
    });

    onDestroy(() => { if (simulation) simulation.stop(); });

    const nodeTypes = ["All","Country","Region","River","Mountain","Place","Concept","Climate","Crop","Industry","Resource","Entity"];
</script>

<svelte:head><title>Knowledge Graph | FutureverseAI</title></svelte:head>

{#if miniMode}
<!-- ── MINI FLOATING PANEL ──────────────────────────────────────────────────── -->
<div
    style="position:fixed; left:{miniX}px; top:{miniY}px; width:{miniW}px; height:{miniH}px;
           z-index:9999; display:flex; flex-direction:column;
           background:#0f172a; border:1px solid #334155; border-radius:12px;
           box-shadow:0 20px 60px rgba(0,0,0,0.6); overflow:hidden;"
>
    <!-- Mini header — drag handle -->
    <div
        on:mousedown={startDrag}
        style="display:flex; align-items:center; justify-content:space-between;
               padding:8px 12px; background:#1e293b; cursor:move; user-select:none;
               border-bottom:1px solid #334155; flex-shrink:0;"
    >
        <div style="display:flex; align-items:center; gap:8px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2">
                <path d="M13 10V3L4 14h7v7l9-11h-7z"/>
            </svg>
            <span style="font-size:12px; font-weight:600; color:#e2e8f0;">Knowledge Graph</span>
            <span style="font-size:10px; color:#64748b; background:#0f172a; padding:1px 6px; border-radius:10px;">
                {stats.nodes} nodes
            </span>
        </div>
        <div style="display:flex; gap:4px;">
            <button on:click={toggleMode}
                style="font-size:10px; background:#334155; color:#94a3b8; border:none;
                       padding:2px 8px; border-radius:4px; cursor:pointer;"
                title="Expand to full screen">⛶</button>
            <button on:click={() => history.back()}
                style="font-size:14px; background:transparent; color:#64748b; border:none;
                       cursor:pointer; padding:0 4px;" title="Close">×</button>
        </div>
    </div>

    <!-- Mini search bar -->
    <div style="display:flex; gap:4px; padding:6px 8px; background:#1e293b;
                border-bottom:1px solid #1e293b; flex-shrink:0;">
        <input
            bind:value={searchQuery}
            on:keydown={(e) => e.key==="Enter" && handleSearch()}
            placeholder="Search entities..."
            style="flex:1; font-size:11px; background:#0f172a; border:1px solid #334155;
                   color:#e2e8f0; border-radius:6px; padding:4px 8px; outline:none;"
        />
        <button on:click={handleSearch}
            style="font-size:11px; background:#4f46e5; color:#fff; border:none;
                   padding:4px 10px; border-radius:6px; cursor:pointer;">Search</button>
        <button on:click={resetSearch}
            style="font-size:11px; background:#334155; color:#94a3b8; border:none;
                   padding:4px 8px; border-radius:6px; cursor:pointer;">Reset</button>
    </div>

    <!-- Mini graph canvas -->
    <div bind:this={container} style="flex:1; position:relative; overflow:hidden;">
        {#if loading}
            <div style="position:absolute; inset:0; display:flex; align-items:center;
                        justify-content:center; background:rgba(15,23,42,0.8);">
                <div style="width:20px; height:20px; border:2px solid #4f46e5;
                             border-top-color:transparent; border-radius:50%; animation:spin 0.8s linear infinite;"></div>
            </div>
        {/if}
        {#if error}
            <div style="position:absolute; inset:0; display:flex; align-items:center;
                        justify-content:center; padding:12px; text-align:center;">
                <span style="font-size:11px; color:#94a3b8;">{error}</span>
            </div>
        {/if}
        <svg bind:this={svgEl} style="width:100%; height:100%;"></svg>
    </div>

    <!-- Mini node detail footer (shows on click) -->
    {#if selectedNode}
        <div style="border-top:1px solid #334155; background:#1e293b; padding:8px 12px;
                    flex-shrink:0; max-height:100px; overflow-y:auto;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <p style="font-size:11px; font-weight:600; color:#e2e8f0; margin:0 0 2px;">
                        {selectedNode.label}
                    </p>
                    <div style="display:flex; gap:6px; flex-wrap:wrap;">
                        <span style="font-size:10px; background:{getColor(selectedNode.type)};
                                     color:#fff; padding:1px 6px; border-radius:10px;">
                            {selectedNode.type}
                        </span>
                        {#if selectedNode.subject}
                            <span style="font-size:10px; color:#94a3b8;">{selectedNode.subject}</span>
                        {/if}
                        {#if selectedNode.chapter}
                            <span style="font-size:10px; color:#64748b;">· {selectedNode.chapter}</span>
                        {/if}
                    </div>
                </div>
                <div style="display:flex; gap:4px; flex-shrink:0;">
                    <button on:click={() => { searchQuery = selectedNode.label; handleSearch(); }}
                        style="font-size:10px; background:#4f46e5; color:#fff; border:none;
                               padding:2px 8px; border-radius:4px; cursor:pointer;">Explore</button>
                    <button on:click={() => selectedNode = null}
                        style="font-size:12px; background:transparent; color:#64748b;
                               border:none; cursor:pointer;">×</button>
                </div>
            </div>
        </div>
    {/if}
</div>

{:else}
<!-- ── FULL SCREEN MODE ────────────────────────────────────────────────────── -->
<div class="flex flex-col h-screen bg-gray-950 text-white overflow-hidden">
    <div class="flex items-center justify-between px-6 py-3 border-b border-gray-800 bg-gray-900 shrink-0">
        <div class="flex items-center gap-3">
            <h1 class="text-lg font-semibold">Knowledge Graph</h1>
            <span class="text-xs text-gray-400 bg-gray-800 px-2 py-0.5 rounded-full">
                {stats.nodes} nodes · {stats.relationships} relationships
            </span>
        </div>
        <div class="flex items-center gap-2">
            <input bind:value={searchQuery}
                on:keydown={(e) => e.key==="Enter" && handleSearch()}
                placeholder="Search entities..."
                class="text-sm bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 w-56 focus:outline-none focus:border-indigo-500"/>
            <button on:click={handleSearch} class="text-sm bg-indigo-600 hover:bg-indigo-500 px-3 py-1.5 rounded-lg">Search</button>
            <button on:click={resetSearch}  class="text-sm bg-gray-800 hover:bg-gray-700 px-3 py-1.5 rounded-lg">Reset</button>
            <button on:click={toggleMode}   class="text-sm bg-gray-700 hover:bg-gray-600 px-3 py-1.5 rounded-lg" title="Float panel">⊡ Mini</button>
        </div>
    </div>
    <div class="flex items-center gap-2 px-6 py-2 border-b border-gray-800 bg-gray-900 shrink-0 overflow-x-auto">
        {#each nodeTypes as type}
            <button on:click={() => filterByType(type)}
                class="text-xs px-3 py-1 rounded-full border transition shrink-0
                    {activeFilter===type ? 'border-indigo-500 bg-indigo-500/20 text-indigo-300' : 'border-gray-700 text-gray-400 hover:border-gray-500'}">
                {#if type !== "All"}<span class="inline-block w-2 h-2 rounded-full mr-1" style="background:{getColor(type)}"></span>{/if}
                {type}
            </button>
        {/each}
    </div>
    <div class="flex flex-1 overflow-hidden">
        <div bind:this={container} class="flex-1 relative overflow-hidden">
            {#if loading}
                <div class="absolute inset-0 flex items-center justify-center bg-gray-950/80 z-10">
                    <div class="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
                </div>
            {/if}
            {#if error}
                <div class="absolute inset-0 flex items-center justify-center">
                    <div class="bg-red-900/30 border border-red-700 rounded-lg p-6 max-w-md text-center">
                        <p class="text-red-400 font-medium mb-2">No results</p>
                        <p class="text-sm text-gray-400">{error}</p>
                        <button on:click={resetSearch} class="mt-3 text-sm bg-gray-700 px-4 py-1.5 rounded-lg">Show all</button>
                    </div>
                </div>
            {/if}
            <svg bind:this={svgEl} class="w-full h-full"></svg>
        </div>
        <div class="w-64 border-l border-gray-800 bg-gray-900 p-4 overflow-y-auto shrink-0">
            {#if selectedNode}
                <div class="flex items-center justify-between mb-3">
                    <h2 class="font-semibold text-sm">Node Details</h2>
                    <button on:click={() => selectedNode=null} class="text-gray-500 hover:text-white text-lg">&times;</button>
                </div>
                <div class="space-y-2">
                    <div><p class="text-xs text-gray-500 mb-1">Name</p><p class="text-sm font-medium break-words">{selectedNode.label}</p></div>
                    <div><p class="text-xs text-gray-500 mb-1">Type</p>
                        <span class="text-xs px-2 py-0.5 rounded-full text-white" style="background:{getColor(selectedNode.type)}">{selectedNode.type}</span>
                    </div>
                    {#if selectedNode.subject}<div><p class="text-xs text-gray-500 mb-1">Subject</p><p class="text-xs text-gray-300">{selectedNode.subject}</p></div>{/if}
                    {#if selectedNode.chapter}<div><p class="text-xs text-gray-500 mb-1">Chapter</p><p class="text-xs text-gray-300">{selectedNode.chapter}</p></div>{/if}
                    {#if selectedNode.doc_name}<div><p class="text-xs text-gray-500 mb-1">Source</p><p class="text-xs text-gray-300 break-words">{selectedNode.doc_name}</p></div>{/if}
                    <div>
                        <p class="text-xs text-gray-500 mb-1">Connections</p>
                        <div class="space-y-1 max-h-40 overflow-y-auto">
                            {#each allEdges.filter(e => String(e.source)===selectedNode.id || String(e.target)===selectedNode.id).slice(0,12) as edge}
                                {@const oid = String(edge.source)===selectedNode.id ? String(edge.target) : String(edge.source)}
                                {@const other = allNodes.find(n=>n.id===oid)}
                                {#if other}
                                    <div class="flex items-center gap-2 text-xs bg-gray-800 rounded px-2 py-1">
                                        <span class="w-2 h-2 rounded-full shrink-0" style="background:{getColor(other.type)}"></span>
                                        <span class="text-gray-300 truncate flex-1">{other.label}</span>
                                        <span class="text-gray-500 text-xs shrink-0">{edge.type.replace("HAS_CONCEPT","→").replace("CONTAINS","∈")}</span>
                                    </div>
                                {/if}
                            {/each}
                        </div>
                    </div>
                    <button on:click={() => { searchQuery=selectedNode.label; handleSearch(); }}
                        class="w-full text-xs bg-indigo-700 hover:bg-indigo-600 px-3 py-1.5 rounded-lg mt-1">
                        Explore connections
                    </button>
                </div>
            {:else}
                <h2 class="font-semibold text-sm mb-3">Legend</h2>
                <div class="space-y-2 mb-4">
                    {#each Object.entries(NODE_COLORS) as [type, color]}
                        <div class="flex items-center gap-2 text-xs text-gray-400">
                            <span class="w-3 h-3 rounded-full shrink-0" style="background:{color}"></span>{type}
                        </div>
                    {/each}
                </div>
            {/if}
        </div>
    </div>
</div>
{/if}

<style>
    @keyframes spin { to { transform: rotate(360deg); } }
</style>