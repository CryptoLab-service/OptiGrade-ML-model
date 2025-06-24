import React from "react";
import x97604931 from "./9760493-1.png";
import x97604933 from "./9760493-3.png";
import { HomeIndicator } from "./HomeIndicator";
import "./style.css";

export const Splashscreen = () => {
  return (
    <div className="splashscreen">
      <div className="overlap-group-wrapper">
        <div className="overlap-group">
          <img className="element" alt="Element" src={x97604933} />

          <img className="img" alt="Element" src={x97604931} />

          <div className="group">
            <div className="logo">
              <div className="group-2" />
            </div>

            <div className="text-wrapper">OptiGrade</div>
          </div>

          <HomeIndicator
            className="home-indicator-instance"
            device="i-phone"
            homeIndicatorClassName="design-component-instance-node"
            orientation="portrait"
          />
          <p className="done">Tap the screen to launch</p>
        </div>
      </div>
    </div>
  );
};
