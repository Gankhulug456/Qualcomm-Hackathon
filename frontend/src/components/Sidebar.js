import React from "react";
import "../css/Sidebar.css";

function Sidebar() {
  return (
    <div className="sidebar">
      <button className="sidebar-button">Home</button>
      <button className="sidebar-button">Recents</button>
      <button className="sidebar-button">Support</button>
    </div>
  );
}

export default Sidebar;
