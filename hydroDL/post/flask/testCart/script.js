var width = 200;
var height = 100;
var hspace = 50;
var vspace = 10;
var nodeId = 0;
var parentLst = new Array();
var leftChildLst = new Array();
var rightChildLst = new Array();
var posLst = new Array();
var levelLst = new Array();

function onClick() {
    var divP = event.target;
    var idP = parseInt(divP.getAttribute('nodeId'));
    pushRow(idP, 'left');
    pushRow(idP, 'right');
    pushCol(idP)
    if (!parentLst.includes(idP)) {
        drawChild(idP, 'left');
        drawChild(idP, 'right');
    }
}

function updateId(idP, direction) {
    if (direction == 'root') {
        parentLst[idP] = -1;
        levelLst[idP] = 0;
        posLst[idP] = 0;
    } else {
        if (direction == 'left') {
            leftChildLst[idP] = nodeId; var lr = 0;
        } else if (direction == 'right') {
            rightChildLst[idP] = nodeId; var lr = 1;
        }
        parentLst[nodeId] = idP;
        levelLst[nodeId] = levelLst[idP] + 1;
        posLst[nodeId] = posLst[idP] * 2 + lr;
    }
    nodeId = nodeId + 1;
    return nodeId - 1
}

function drawChild(idP, direction) {
    var idC = nodeId
    var div = document.createElement('div');
    if (direction == 'root') {
        var leftPx = 200;
        var topPx = vspace
    } else {
        var divP = document.getElementById('div-' + idP)
        var divArrow = document.createElement('div');
        var centerPx = parseInt(divP.style.left) + width / 2
        if (direction == 'left') {
            var leftPx = centerPx - width - vspace;
            divArrow.setAttribute('class', 'arrowLeft');
            var LeftArrow = leftPx + width / 2
        } else if (direction == 'right') {
            var leftPx = centerPx + vspace;
            divArrow.setAttribute('class', 'arrowRight');
            var LeftArrow = centerPx;
        }
        var topPx = parseInt(divP.style.top) + height + hspace;
        divArrow.setAttribute('nodeId', parseInt(nodeId));
        divArrow.setAttribute('id', 'arrow-' + nodeId)
        divArrow.style.left = LeftArrow + 'px';
        divArrow.style.top = parseInt(divP.style.top) + height + 'px';
        document.body.appendChild(divArrow);
    }
    div.setAttribute('class', 'node');
    div.setAttribute('nodeId', parseInt(nodeId));
    div.setAttribute('id', 'div-' + nodeId);
    div.style.left = leftPx + 'px';
    div.style.top = topPx + 'px';
    div.innerText = nodeId
    div.onclick = function () { onClick(); };
    document.body.appendChild(div);
    idC = updateId(idP, direction)
    pushRow(idC, direction);
    return div
}

function pushRow(idC, direction) {
    var idS = findSideNode(idC, direction);
    var divP = document.getElementById('div-' + idC);
    var leftPx = parseInt(divP.style.left);
    var leftNew = leftPx
    for (var k = 0; k < idS.length; k++) {
        var div = document.getElementById('div-' + idS[k]);
        if (direction == 'left') {
            leftNew = leftPx - width - vspace * 2;
        } else if (direction == 'right') {
            leftNew = leftPx + width + vspace * 2;
        }
        var leftOld = parseInt(div.style.left);
        var offset = leftNew - leftOld;
        div.style.left = leftNew + 'px';
        leftPx = leftNew
        if (offset !== 0) {
            moveArrow(idS[k], offset)
        }
    }
    if (leftNew < vspace) {
        moveAllRight(vspace - leftNew)
    }
}

function pushCol(idP) {
    var divP = document.getElementById('div-' + idP)
    var leftPx = parseInt(divP.style.left)
    while (parentLst[idP] !== -1) {
        var idG = parentLst[idP]
        var divG = document.getElementById('div-' + idG)
        var leftOld = parseInt(divG.style.left);
        if (leftChildLst.includes(idP)) {
            divG.style.left = leftPx + width / 2 + vspace + 'px'
        } else if (rightChildLst.includes(idP)) {
            divG.style.left = leftPx - width / 2 - vspace + 'px'
        }
        offset = parseInt(divG.style.left) - leftOld;
        if (offset !== 0) {
            moveArrow(idG, offset)
        }
        pushRow(idG, 'left')
        pushRow(idG, 'right')
        divP = divG
        idP = idG
        leftPx = parseInt(divP.style.left)
    }
}

function moveArrow(idC, offset) {
    var id0 = idC;
    if (idC !== 0) {
        var arrow0 = document.getElementById('arrow-' + id0);
        moveArrowBottom(arrow0, offset);
    }

    var id1 = leftChildLst[idC];
    if (id1 !== undefined) {
        var arrow1 = document.getElementById('arrow-' + id1);
        moveArrowTop(arrow1, offset);
    }
    var id2 = rightChildLst[idC];
    if (id2 !== undefined) {
        var arrow2 = document.getElementById('arrow-' + id2);
        moveArrowTop(arrow2, offset);
    }
}

function moveArrowBottom(arrow, offset) {
    if (arrow.className == 'arrowLeft') {
        var newWidth = arrow.offsetWidth - offset;
        if (newWidth > 0) {
            arrow.style.width = newWidth + 'px';
            arrow.style.left = parseInt(arrow.style.left) + offset + 'px';
        } else {
            arrow.setAttribute('class', 'arrowRight');
            arrow.style.left = parseInt(arrow.style.left) + arrow.offsetWidth + 'px';
            arrow.style.width = -newWidth + 'px';
        }
    } else if (arrow.className == 'arrowRight') {
        var newWidth = arrow.offsetWidth + offset;
        if (newWidth > 0) {
            arrow.style.width = newWidth + 'px';
        } else {
            arrow.setAttribute('class', 'arrowLeft');
            arrow.style.width = -newWidth + 'px';
            arrow.style.left = parseInt(arrow.style.left) + newWidth + 'px';
        }
    }
}

function moveArrowTop(arrow, offset) {
    if (arrow.className == 'arrowLeft') {
        var newWidth = arrow.offsetWidth + offset
        if (newWidth > 0) {
            arrow.style.width = newWidth + 'px';
        } else {
            arrow.setAttribute('class', 'arrowRight');
            arrow.style.width = -newWidth + 'px';
            arrow.style.left = parseInt(arrow.style.left) + newWidth + 'px';
        }
    } else if (arrow.className == 'arrowRight') {
        var newWidth = arrow.offsetWidth - offset;
        if (newWidth > 0) {
            arrow.style.width = newWidth + 'px';
            arrow.style.left = parseInt(arrow.style.left) + offset + 'px';
        } else {
            arrow.setAttribute('class', 'arrowLeft');
            arrow.style.left = parseInt(arrow.style.left) + arrow.offsetWidth + 'px';
            arrow.style.width = -newWidth + 'px';

        }
    }
}

function moveAllRight(offset) {
    var divLst = document.getElementsByClassName('node');
    for (var div of divLst) {
        div.style.left = parseInt(div.style.left) + offset + 'px'
    }
    var divLst = document.getElementsByClassName('arrowLeft');
    for (var div of divLst) {
        div.style.left = parseInt(div.style.left) + offset + 'px'
    }
    var divLst = document.getElementsByClassName('arrowRight');
    for (var div of divLst) {
        div.style.left = parseInt(div.style.left) + offset + 'px'
    }
}

function arrayIndex(array, search, except = [-1]) {
    var i = array.indexOf(search),
        indexes = [];
    while (i !== -1) {
        if (!except.includes(i)) {
            indexes.push(i);
        }
        i = array.indexOf(search, ++i);
    }
    return indexes;
}

function findSideNode(idC, direction) {
    var idTemp = arrayIndex(levelLst, levelLst[idC], [idC]);
    var idOut = new Array();
    var distOut = new Array();
    for (var k = 0; k < idTemp.length; k++) {
        value = idTemp[k]
        if ((posLst[value] < posLst[idC] && direction == 'left') || (posLst[value] > posLst[idC] && direction == 'right')) {
            idOut.push(parseInt(value));
            distOut.push(Math.abs(posLst[value] - posLst[idC]));
        }
    }
    sortOut = sortTwoArray(distOut, idOut)
    return sortOut[1]
}

function sortTwoArray(aryRef, arySrt) {
    var list = [];
    for (var j = 0; j < aryRef.length; j++)
        list.push({ 'ref': aryRef[j], 'srt': arySrt[j] });
    list.sort(function (a, b) {
        return ((a.ref < b.ref) ? -1 : ((a.ref == b.ref) ? 0 : 1));
    });
    for (var k = 0; k < list.length; k++) {
        aryRef[k] = list[k].ref;
        arySrt[k] = list[k].srt;
    }
    return [aryRef, arySrt]
}