// $.ajax({
//     url: "",
//     type: "POST",
//     data: "Hi!",
//     processData: false,
//     contentType: false,
//     success: function (response) {
//         console.log("response: " + response);
//     },
//     error: function (jqXHR, textStatus, errorMessage) {
//         console.log(errorMessage);
//     }
// })

// Nav tab selection renderer
let nav_bar_tabs = document.getElementById("nav_bar")
    .getElementsByTagName("ul").item(0)
    .getElementsByTagName("li");
let contents = document.getElementById("main-content")
    .getElementsByClassName("page-content");

for (let i = 0; i < nav_bar_tabs.length; i++) {
    nav_bar_tabs.item(i).onclick = function () {
        for (let j = 0; j < nav_bar_tabs.length; j++) {
            if (j != i) {
                nav_bar_tabs.item(j).classList.remove("active");
                contents.item(j).classList.remove("active");
            }
        }
        nav_bar_tabs.item(i).classList.add("active");
        contents.item(i).classList.add("active")
    }
}

// Dropdown list animations



// let scd_options = [];
// const scd_arrows_containers = document.getElementsByClassName("scd-pointer");
// const scd_containers = document.getElementsByClassName("select-config-dropdown-container");
// const scd_lists = document.getElementsByClassName("select-config-dropdown")

// function updateAnimations() {
//     for (let i = 0; i < scd_containers.length; i++) {
//         scd_containers.item(i).onmouseover = function () {
//             if (scd_options.length == 0) {
//                 setSCDArrowState("arrow-plus", scd_containers.item(i).getElementsByClassName("scd-pointer").item(0));
//             } else {
//                 setSCDArrowState("arrow-down", scd_containers.item(i).getElementsByClassName("scd-pointer").item(0));
//             }
//         }
//         scd_containers.item(i).onmouseout = function () {
//             setSCDArrowState("arrow-left", scd_containers.item(i).getElementsByClassName("scd-pointer").item(0));
//         }
//     }
// }

// function setSCDArrowState(state, container) {
//     const availableClasses = ["arrow-down", "arrow-left", "arrow-plus"];
//     for (let i = 0; i < availableClasses.length; i++) {
//         if (state != availableClasses[i]) {
//             container.classList.remove(availableClasses[i]);
//         } else {
//             container.classList.add(availableClasses[i]);
//         }
//     }
// }

// function toggleDropdown(dp_container, arrow_conatiner=undefined, toggle_class = "", available_classes = []) {
//     const ul = dp_container.getElementsByClassName("dp-content").item(0);
//     const ul_height = getComputedStyle(ul).height;
//     if (dp_container.classList.contains("dp-opnd")) {
//         console.log(toggle_class);
//         dp_container.classList.remove("dp-opnd");
//         if (toggle_class != "") {
//             // for (let i = 0; i < available_classes.length; i++) {
//             //     dp_container.classList.remove(available_classes[i]);
//             // }
//             dp_container.classList.add(toggle_class);
//         }
//         dp_container.removeAttribute("style");
//         arrow_conatiner.removeAttribute("style");
//     } else {
//         dp_container.classList.add("dp-opnd");
//         dp_container.setAttribute("style", "height: " + ul_height);
//         dp_container.onclick = undefined;
//     }
// }

// function updateDropdownFunctions(dp_container) {
//     const lis = dp_container.getElementsByTagName("li");
//     const arrow_conatiner = dp_container.getElementsByClassName("dp-pointer").item(0);
//     let available_classes = [];
//     for (let i = 0; i < lis.length; i++) {
//         const dp_cls_name = lis.item(i).getAttribute("dp_cls_name");
//         available_classes = available_classes.concat(dp_cls_name);
//         const li_height = parseInt(getComputedStyle(lis.item(i)).height.substring(-2)) + parseInt(getComputedStyle(lis.item(i)).paddingTop.substring(-2)) + parseInt(getComputedStyle(lis.item(i)).paddingBottom.substring(-2));
//         lis.item(i).onmouseover = function(){
//             arrow_conatiner.setAttribute('style', 'transform:translateY(' + i * li_height + 'px);')
//         }
//     }
//     console.log(available_classes);
//     for(let i = 0; i < lis.length; i++){
//         const dp_cls_name = lis.item(i).getAttribute("dp_cls_name");
//         lis.item(0).onclick = toggleDropdown(dp_container, arrow_conatiner, dp_cls_name, available_classes);
//     }
// }

// function updateSCD() {
//     if (scd_options.length == 0) {
//         for (let i = 0; i < scd_containers.length; i++) {
//             scd_containers.item(i).onclick = function () {
//                 console.log("call add model");
//                 scd_options = scd_options.concat({ type: "model1", name: "option1" });
//                 console.log(scd_options)
//                 updateSCDList();
//             }
//         }
//     } else {
//         for (let i = 0; i < scd_containers.length; i++) {
//             scd_containers.item(i).onclick = function () {
//                 console.log("show full list");
//                 toggleDropdown(scd_containers.item(i));
//             }
//         }
//     }
// }

// function updateSCDList() {
//     for (let i = 0; i < scd_lists.length; i++) {
//         let inner_ul = '<li dp_cls_name="add model">ADD MODEL</li>';
//         for (let j = 0; j < scd_options.length; j++) {
//             inner_ul += '<li dp_cls_name="' + scd_options[j].name.toLowerCase() + '">' + scd_options[j].name.toUpperCase() + '(' + scd_options[j].type.toUpperCase() + ')' + '</li>';
//         }
//         inner_ul += '<li dp_cls_name="custom settings">CUSTOM SETTINGS</li>'
//         scd_lists.item(i).innerHTML = inner_ul;
//     }
//     for (let i = 0; i < scd_containers.length; i++) {
//         const add_styles = scd_containers.item(i).getElementsByClassName("dp-add-styles").item(0);
//         const li_el = scd_containers.item(i).getElementsByTagName("li").item(0);
//         const li_height = parseInt(getComputedStyle(li_el).height.substring(-2)) + parseInt(getComputedStyle(li_el).paddingTop.substring(-2)) + parseInt(getComputedStyle(li_el).paddingBottom.substring(-2));
//         // add_styles.innerHTML = scd_containers.item(0).className
//         let inner_styles = ''
//         for (let j = 0; j < scd_options.length; j++) {
//             inner_styles += "." + scd_containers.item(0).className + "." + scd_options[j].name.toLowerCase() + ' ul{transform: translateY(-' + (j + 1) * li_height + 'px);}';
//         }
//         add_styles.innerHTML = inner_styles;
//         updateDropdownFunctions(scd_containers.item(i));
//     }
//     updateSCD();
// }

// updateAnimations();
// updateSCD();