import { mount } from "svelte";
import "./app.css";
import App from "./App.svelte";
import { app as impsy } from "./lib/appState.svelte";

const app = mount(App, { target: document.getElementById("app")! });

// Load the bundled demo model in the background so the app is ready to play
// the moment it opens; the user can still load their own .tflite over it.
void impsy.autoLoadDefaultModel();

export default app;
