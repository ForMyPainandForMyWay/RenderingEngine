import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// 导入自定义组件
import "components" as Components
import "views" as Views
import "widgets" as Widgets

ApplicationWindow {
    width: 640
    height: 480
    visible: true
    color: "lightgray"
    // 顶部选项栏
    menuBar: Widgets.TopMenuBar{}

    // 渲染展示区
    Rectangle {
        width: 400
        height: 400
        color: "gray"
        anchors.verticalCenter: parent.verticalCenter
        anchors.left: parent.left
        anchors.leftMargin: 10
    }

    // 右侧标签页
    ColumnLayout  {
        width: parent.width * 0.3
        height: parent.height * 0.9
        anchors.right: parent.right
        anchors.rightMargin: 20

        TabBar {
            id: bar
            spacing: 5
            Layout.fillWidth: true
            onCurrentIndexChanged: stack.currentIndex = currentIndex
            background: Rectangle {
                id: tabBarRect
                color: "transparent"
            }
            TabButton {
                text: "Lights"
                width: 60
                height: 20
                anchors.bottom: parent.bottom
                background: Rectangle {
                    radius: tablePage.radius
                    color: {
                        if (parent.checked) return "white";
                        return "gray";
                    }
                }
                contentItem: Label {
                    text: parent.text
                    color: parent.checked ? "black" : "white"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    font: parent.font
                }
            }
            TabButton {
                text: "Model"
                width: 60
                height: 20
                anchors.bottom: parent.bottom
                background: Rectangle {
                    radius: tablePage.radius
                    color: {
                        if (parent.checked) return "white";
                        return "gray";
                    }
                }
                contentItem: Label {
                    text: parent.text
                    color: parent.checked ? "black" : "white"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    font: parent.font
                }
            }
        }

        Rectangle {
            id: tablePage
            Layout.fillWidth: true
            Layout.fillHeight: true
            border.width: 2
            border.color: "black"
            radius: 10
            StackLayout {
                id: stack
                anchors.fill: parent
                anchors.margins: 10  // 避免覆盖选项卡背景
                // Light选项
                Rectangle {
                    radius: tablePage.radius
                    property var colorRecRate: 0.3
                    // Color栏
                    Components.Block {
                        boxTitle: "Color"
                        boxCornerRadius: tablePage.radius
                        boxHeightRatio: parent.colorRecRate
                        pTop: parent.top
                        Views.ColorBlockColumn{}  // 灯光颜色调整栏
                    }
                    // 位置栏
                    Components.Block{
                        boxTitle: "Position"
                        boxCornerRadius: tablePage.radius
                        boxHeightRatio: 1 - parent.colorRecRate - 0.02
                        pBottom: parent.bottom
                        Views.PosiBlockColumn{}  // 聚光灯位置调整
                    }
                }
                // Model选项
                Rectangle {
                    radius: tablePage.radius
                    property var colorRecRate: 0.2
                    // Mesh栏
                    Components.Block {
                        boxTitle: "Mesh"
                        boxCornerRadius: tablePage.radius
                        boxHeightRatio: parent.colorRecRate
                        pTop: parent.top
                        Views.MeshBlockColumn{}

                    }
                    // Camera栏
                    Components.Block{
                        boxTitle: "Camera"
                        boxCornerRadius: tablePage.radius
                        boxHeightRatio: 1 - parent.colorRecRate - 0.02
                        pBottom: parent.bottom
                        Views.CameraBlockColumn{}  // 相机参数调整
                    }
                }
            }
        }
    }
}