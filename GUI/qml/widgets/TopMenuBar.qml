import QtQuick.Controls
import QtQuick.Dialogs

MenuBar {
    // 打开模型
    Menu {
        title: "File"
        MenuItem {
            text: "Open"
            onTriggered: { fileOpenDialog.open() }
            FileDialog {
                id: fileOpenDialog
                title: "选择模型文件"
                nameFilters: ["模型文件 (*.obj)"]
                fileMode: FileDialog.OpenFile
                onAccepted: {
                    settingProxy.openObj(fileOpenDialog.currentFile)
                }
            }
        }
        MenuItem {
            text: "Save"
            onTriggered: { fileSaveDialog.open() }
            FileDialog {
                id: fileSaveDialog
                title: "保存渲染图片文件"
                nameFilters: ["(*.png)", "(*.jpg)"]
                fileMode: FileDialog.SaveFile
                onAccepted: {
                    settingProxy.saveImg(fileSaveDialog.currentFile)
                }
            }
        }
    }
    // 设置
    Menu {
        title: "Settings"
        // 渲染模式
        Menu {
            title: "Render modes"
            ActionGroup {
                id: piplineOpt
                exclusive: true
                onCheckedActionChanged: {
                    if (checkedAction) {
                        settingProxy.setRenderMode(checkedAction.isRaster)
                    }
                }
                Action {
                    text: "Rasterization"
                    id: rasAction
                    property bool isRaster: true
                    checkable: true
                    checked: true
                }
                Action {
                    text: "Path Tracing"
                    property bool isRaster: false
                    id: rtAction
                    checkable: true
                }
            }
            MenuItem {
                action: rasAction
            }
            MenuItem {
                action: rtAction
            }
        }
        // 引擎效果设置
        Menu {
            title: "Rendering Settings"
            MenuItem {
                text: "Enable AO"
                checkable: true
                checked: false
                onCheckedChanged: {
                    settingProxy.enableSSAO(checked)
                }
            }
            MenuItem {
                text: "Enable ShadowMapping"
                checkable: true
                checked: false
                onCheckedChanged: {
                    settingProxy.enableShadow(checked)
                }
            }
            MenuItem {
                text: "Enable SkyBox"
                checkable: true
                checked: false
                onCheckedChanged: {
                    settingProxy.enableSkyBox(checked)
                }
            }
            Menu {
                title: "FXAA Strength"
                ActionGroup {
                    id: aaOptions
                    exclusive: true
                    onCheckedActionChanged: {
                        if (checkedAction) {
                            settingProxy.setFXAA(checkedAction.level)
                        }
                    }
                    Action {
                        text: "High"
                        id: highAction
                        property int level: 3
                        checkable: true
                    }
                    Action {
                        text: "Medium"
                        id: mediumAction
                        property int level: 2
                        checkable: true
                    }
                    Action {
                        text: "Low"
                        id: lowAction
                        property int level: 1
                        checkable: true
                    }
                    Action {
                        text: "Off"
                        id: offAction
                        property int level: 0
                        checkable: true
                        checked: true
                    }
                }
                MenuItem {
                    action: highAction
                }
                MenuItem {
                    action: mediumAction
                }
                MenuItem {
                    action: lowAction
                }
                MenuItem {
                    action: offAction
                }
            }
        }
    }
}