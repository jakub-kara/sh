function get_span() {
    spans = Array.prototype.slice.call(document.getElementsByClassName("tab"));
    for (var span of spans) {
        create_table(span);
    }
}

function create_table(el) {
    typ = el.hasAttribute("data-type") ? el.getAttribute("data-type") : "any";
    def = el.hasAttribute("data-default") ? el.getAttribute("data-default") : "";
    opt = el.hasAttribute("data-options") ? JSON.parse(el.getAttribute("data-options")) : {};

    tab = document.createElement("table");
    tr = document.createElement("tr");
    td = document.createElement("td");
    td.innerHTML = "Type";
    tr.appendChild(td);
    td = document.createElement("td");
    td.innerHTML = typ;
    tr.appendChild(td);
    tab.appendChild(tr);

    tr = document.createElement("tr");
    td = document.createElement("td");
    td.innerHTML = "Default";
    tr.appendChild(td);
    td = document.createElement("td");
    td.innerHTML = def;
    tr.appendChild(td);
    tab.appendChild(tr);

    i = 0;
    for (var prop in opt) {
        tr = document.createElement("tr");
        if (i == 0) {
            td = document.createElement("td");
            td.innerHTML = "Options";
            if (Object.keys(opt).length > 1) {
                td.setAttribute("rowspan", "0")
            }
            tr.appendChild(td);
        }

        td = document.createElement("td");
        td.innerHTML = prop + "<br>" + opt[prop];
        tr.appendChild(td);
        tab.appendChild(tr);
        i++;
    }

    par = el.parentElement
    par.replaceChild(tab, el);
}

function expand_sidebar() {
    document.getElementById("main").style.marginLeft = "155px";
}

function collapse_sidebar() {
    document.getElementById("main").style.marginLeft = "55px";
}