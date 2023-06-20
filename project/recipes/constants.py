STOPWORDS = {'the', 'a', 'an', 'of', 'is', 'was'}
ALL_ARGS = ['arg0', 'arg1', 'argL', 'argT']
ROLESET_ID = 'roleset_id'
HTML_INPUT = """
    <style>
        .c0178m {
          box-sizing: border-box;
        }
        .c0179m {
          width: 100%;
          font-size: 20px;
          word-wrap: break-word;
          white-space: normal;
        }
        .c0179m:focus {
          outline: 0;
        }
        .c0179m:empty {
          display: none;
        }
        .c0179m iframe, .c0179 img {
          maxwidth: 100%;
        }
        .c0180m {
          padding: 20px;
          padding: 20px;
          text-align: center;
        }
        .c01131m {
            border: 1px solid #ddd;
            text-align: left;
            border-radius: 4px;
        }
        .c01131m:focus-within {
          box-shadow: 0 0 0 1px #583fcf;
          border-color: #583fcf;
        }
        .c01132m {
          top: -3px;
          opacity: 0.75;
          position: relative;
          font-size: 12px;
          font-weight: bold;
          padding-left: 10px;
        }
        .c01133m {
          width: 100%;
          border: 0;
          padding: 10px;
          font-size: 20px;
          background: transparent;
          font-family: "Lato", "Trebuchet MS", Roboto, Helvetica, Arial, sans-serif;
        }
        .c01133m:focus {
          outline: 0;
        }
    </style>
    <form method="GET" id="my_form"></form>
    <div class="c01131m">
    <label class="c01132m" for="event_descriptor_table">Event Descriptor</label>
    <table style="white-space:nowrap;width:100%;" id="event_descriptor_table">
        <tr>
            <td id='td_roleset' width="25%">
                <div class="prodigy-content c0178m c0179m c0180m" style="padding: 0px;" >
                    <div class="c01131m" >
                        <label class="c01132m" for="roleset">roleset_id</label>
                        <input type="text" id="roleset" form="my_form" placeholder="roleset_id..." 
                        class="prodigy-text-input c01133m" tabindex="1" name="roleset"
                        onchange="updateRoleset(this.value)"  list="rolesetList"
                        style="font-size: 12pt;" />
                    </div>
                </div>
                <datalist id="rolesetList" name="rolesetList"/>
            </td>
            <td >
                <div class="prodigy-content c0178m c0179m c0180m" style="padding: 0px;">
                    <div class="c01131m" style="padding: 0x">
                        <label class="c01132m" for="arg0">ARG-0</label>
                        <input type="text" id="arg0" form="my_form" placeholder="ARG-0..." 
                        class="prodigy-text-input c01133m" tabindex="1" name="arg0"
                        onchange="updateArg0(this.value)"  list="arg0List"
                        style="font-size: 12pt;"/>
                    </div>
                </div>
            	<datalist id="arg0List" name="arg0List"/>
            </td>
            <td >
                <div class="prodigy-content c0178m c0179m c0180m" style="padding: 0px;">
                    <div class="c01131m" style="padding: 0x">
                        <label class="c01132m" for="arg1">ARG-1</label>
                        <input type="text" id="arg1" form="my_form" placeholder="ARG-1..." 
                        class="prodigy-text-input c01133m" tabindex="1" name="arg1"
                        onchange="updateArg1(this.value)"  list="arg1List"
                        style="font-size: 12pt;"/>
                    </div>
                </div>
            	<datalist id="arg1List" name="arg1List"/>
            </td>
        <tr>
        <tr>
            <td/>
            <td >
                <div class="prodigy-content c0178m c0179m c0180m" style="padding: 0px;">
                    <div class="c01131m" style="padding: 0x">
                        <label class="c01132m" for="argL">ARG-L</label>
                        <input type="text" id="argL" form="my_form" placeholder="ARG-L..." 
                        class="prodigy-text-input c01133m" tabindex="1"  name="argL"
                        onchange="updateArgL(this.value)"  list="argLList"
                        style="font-size: 12pt;"/>
                    </div>
                </div>
            	<datalist id="argLList" name="argLList"/>
            </td>
            <td >
                <div class="prodigy-content c0178m c0179m c0180m" style="padding: 0px;">
                    <div class="c01131m" style="padding: 0x">
                        <label class="c01132m" for="argT">ARG-T</label>
                        <input type="text" id="argT" form="my_form" placeholder="ARG-T..." 
                        class="prodigy-text-input c01133m" tabindex="1" name="argT"
                        onchange="updateArgT(this.value)"  list="argTList"
                        style="font-size: 12pt;"/>
                    </div>
                </div>
            	<datalist id="argTList" name="argTList"/>
            </td>
        </tr>
    </table>
    </div>    
    <style onload="doThis()" />
        """
JAVASCRIPT_WSD = """
    let allRolesets = []
    let allArg0s = []
    let allArg1s = []
    let allArgLs = []
    let allArgTs = []
    let prevTaskHash = null
    let testVar = null

    function createOption(ddl, text, value) {
        var opt = document.createElement('option');
        opt.value = value;
        opt.text = text;
        ddl.options.add(opt);
    }

    function createOptions(ddl, suggestions, ddlArray) {
        ddlArray = [...new Set([...ddlArray, ...suggestions])]
        ddlArray.sort()
        const options = ddlArray.map(user_input => `<option value="${user_input}" />`)
        ddl.innerHTML = options.join('')
    }

    document.addEventListener('prodigyanswer', event => {
        // This runs every time the user submits an annotation
        const { answer, task } = event.detail
        // console.log('The answer was: ', answer)
        console.log('The answer was: ', roleset.value)
        // Update the rolesets with a unique list of previous + current
        allRolesets = [...new Set([...allRolesets, task.roleset_id, ...task.field_suggestions.roleset_id])]
        allArg0s = [...new Set([...allArg0s, task.arg0, ...task.field_suggestions.arg0])]
        allArg1s = [...new Set([...allArg1s, task.arg1, ...task.field_suggestions.arg1])]
        allArgLs = [...new Set([...allArgLs, task.argL, ...task.field_suggestions.argL])]
        allArgTs = [...new Set([...allArgTs, task.argT, ...task.field_suggestions.argT])]

    })

    document.addEventListener('prodigyupdate', event => {
        // This runs every time the task is updated
        const { task } = event.detail
        if (prevTaskHash !== task._task_hash) {  // we have a new example
            console.log('The cool answer was: ', roleset.value)
            createOptions(rolesetList, task.field_suggestions.roleset_id, allRolesets)
            createOptions(arg0List, task.field_suggestions.arg0, allArg0s)
            createOptions(arg1List, task.field_suggestions.arg1, allArg1s)
            createOptions(argLList, task.field_suggestions.argL, allArgLs)
            createOptions(argTList, task.field_suggestions.argT, allArgTs)
            prevTaskHash = task._task_hash

            arg0.value = task.arg0
            arg1.value = task.arg1
            argL.value = task.argL
            argT.value = task.argT
            if (arg0.disabled) {
                setFocusToTextBox(task.roleset_id)
            }
            else {
                roleset.value = task.roleset_id;
                roleset.disabled = true;
                arg0.scrollIntoView();
                arg0.focus();
                arg0.value = '';
                arg0.value = task.arg0; //set that value back. 
            }
        } else {
        console.log('The un cool answer was: ', roleset.value)
            // createOptions(document.getElementById("rolesetList"), task.roleset_suggestions, allRolesets)
        }
    })

    function setFocusToTextBox(val) {
        roleset.scrollIntoView();
        roleset.focus();
        roleset.value = val; //set that value back. 
    }

    function updateRoleset(val) {
        console.log('updating role')
        window.prodigy.update({roleset_id: val})
    }

    function updateArg0(val) {
        window.prodigy.update({arg0: val})
    }

    function updateArg1(val) {
        window.prodigy.update({arg1: val})
    }

    function updateArgL(val) {
        window.prodigy.update({argL: val})
    }

    function updateArgT(val) {
        window.prodigy.update({argT: val})
    }

    function scrollToMark() {
        document.getElementById("mark_id").scrollIntoView();
    }

    """

DO_THIS_JS = """
function doThis() {
        roleset.style.backgroundColor = "#CCCCCC";
        window.prodigy.update({});
    }
"""

DO_THIS_JS_DISABLE = """
function doThis() {
        arg0.disabled = true;
        arg1.disabled = true
        argL.disabled = true;
        argT.disabled = true;
        arg0.style.backgroundColor = "#CCCCCC";
        arg1.style.backgroundColor = "#CCCCCC";
        argL.style.backgroundColor = "#CCCCCC";
        argT.style.backgroundColor = "#CCCCCC";
        window.prodigy.update({});
    }
"""

PB_HTML = """
                <div class="c01131m">
                <label class="c01132m" for="propBankSite">PropBank</label>
                <iframe src = "{{prop_holder}}" 
                    width='100%' 
                    height='400' 
                    scrolling='yes' 
                    frameborder='no' 
                    id="propBankSite"> 
                </iframe>
                </div>
                """
DOC_HTML = """
                <div class="c01131m">
                <label class="c01132m" for="documentSite">Document: {{doc_id}}</label>
                <iframe src = "{{doc_host}}" 
                    width='100%' 
                    height='200' 
                    scrolling='yes' 
                    frameborder='no' 
                    id="documentSite"> 
                </iframe>
                </div>
                """

# DOC_HTML2 = """
# <div class="myBox" height='500' >
#     {{bert_doc}}
# </div>
#
# """

DOC_HTML2 = """
<style>
    .myBox {
        font-size: 14px;
        border: none;
        padding: 20px;
        width: 100%;
        height: 200px;
        overflow: scroll;
        line-height: 1.5;
    }
    p {
    margin: -15px 10px;
}
</style>
<div class="c01131m">
<label class="c01132m" for="documentSite">Document: {{doc_id}}</label>
<body>
<div class="myBox">
<p>
    {{{bert_doc}}}
<div>
</body>
</div>
<style onload="scrollToMark()" />
<script>
function scrollToMark() {
        console.log('Hello World');
        document.getElementById({{mention_id}}).scrollIntoView();
    }
</script>
"""