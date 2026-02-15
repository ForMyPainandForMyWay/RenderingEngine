import QtQuick

Rectangle {
    id: root
    // ===== 可配置属性（提升复用性）=====
    property string boxTitle: "Color"      // 标题文本
    property real titleFontSize: 12        // 标题字号
    property color titleColor: Qt.rgba(0,0,0,1)    // 标题颜色
    property real boxHeightRatio: 0.3      // 高度占父容器比例
    property real boxCornerRadius: 10      // 圆角
    property color boxBorderColor: Qt.rgba(0,0,0,1)  // 边框颜色
    property color titleBgColor: Qt.rgba(1,1,1,1)    // 标题背景色
    property int boxBorderWidth: 2
    property var pTop: undefined;  // 父容器的顶部
    property var pBottom: undefined;

    // ===== 布局与样式 =====
    width: parent ? parent.width : 200
    height: parent && parent.height ? parent.height * boxHeightRatio : 100
    anchors.top: pTop ? pTop : undefined
    anchors.bottom: pBottom ? pBottom : undefined
    border.width: boxBorderWidth
    border.color: boxBorderColor
    radius: boxCornerRadius
    color: "transparent" // 避免遮挡内容

    // ===== 标题区域（固定顶部）=====
    Text {
        id: titleText
        text: root.boxTitle
        color: root.titleColor
        font.bold: true
        font.pixelSize: root.titleFontSize
        anchors {
            top: parent.top
            topMargin: -contentHeight / 2
            left: parent.left
            leftMargin: 15
        }
        // 标题背景（确保在文字下方）
        Rectangle {
            anchors.fill: parent
            color: root.titleBgColor
            z: -1
        }
    }

    // ===== 内容区域支持外部添加控件=====
    Item {
        id: contentArea
        anchors {
            top: titleText.bottom
            topMargin: 8
            left: parent.left; right: parent.right
            bottom: parent.bottom
            margins: 10 // 内边距，避免贴边
        }
    }
    default property alias content: contentArea.children
}