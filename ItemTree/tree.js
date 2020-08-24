// D3 Setup
var svg = d3.select("svg"),
width = +svg.attr("width"),
height = +svg.attr("height");

var color = d3.scaleOrdinal(d3.schemeCategory10);

var simulation = d3.forceSimulation()
.force("link", d3.forceLink().id(function(d) { return d.id; }))
.force("charge", d3.forceManyBody()
.distanceMax(1000)
.strength(-500)
.distanceMin(30))
.force("center", d3.forceCenter(width / 2, height / 2));

function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(1.0).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
}

function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

var g = svg.append("g").attr("class", "everything");

//Zoom functions 
function zoom_actions(){
    g.attr("transform", d3.event.transform)
}

var zoom_handler = d3.zoom()
    .on("zoom", zoom_actions);



// Function for rendering the graph once we receive it
function renderData(graph) {
    zoom_handler(svg)
    
    var link = g.append("g")
    .attr("class", "links")
    .selectAll("line")
    .data(graph.links)
    .enter().append("line")
    .attr("stroke-width", function(d) { return Math.sqrt(d.value); });
    
    var node = g.append("g")
    .attr("class", "nodes")
    .selectAll("g")
    .data(graph.nodes)
    .enter().append("g")
    
    var circles = node.append("circle")
    .attr("r", 5)
    .attr("fill", function(d) { return color(d.group); })
    .call(d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended));
    
    var lables = node.append("text")
    .text(function(d) {
        return d.id;
    })
    .attr('x', 6)
    .attr('y', 3);
    
    node.append("title")
    .text(function(d) { return d.id; });
    
    simulation
    .nodes(graph.nodes)
    .on("tick", ticked);
    
    simulation.force("link")
    .links(graph.links);
    
    function ticked() {
        link
        .attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });
        
        node
        .attr("transform", function(d) {
            return "translate(" + d.x + "," + d.y + ")";
        })
    }
};

// Extract the data from the API
let url = "http://localhost:8000/query"
d3.json(url)
.then(function(d) {
    renderData(d);
})
.catch(function(err){
    console.log(err);
});
