<template>
  <v-app id="inspire">


    <v-app-bar
        app
        clipped-right
        height="72"
    >
      <v-responsive max-width="300">
        <v-file-input
            style="margin-top: 25px;"
            label="Input Files"
            outlined
            v-on:change="sendFilepath"
        ></v-file-input>
      </v-responsive>
      <v-spacer></v-spacer>
    </v-app-bar>


    <v-navigation-drawer
        app
        clipped
        right
    >
      <v-list>
        <v-list-item
            v-for="n in 5"
            :key="n"
            link
        >
          <v-list-item-content>
            <v-list-item-title>Item {{ n }}</v-list-item-title>
          </v-list-item-content>
        </v-list-item>
      </v-list>
    </v-navigation-drawer>

    <v-main style="margin-top: 10px">
      <div style="position: relative">
        <canvas v-if="channel_view_1" id="layer0" width="600" height="400"
                style="position: absolute; top:25px; left:25px; z-index: 0;"></canvas>
        <canvas v-if="channel_view_2" id="layer1" width="600" height="400"
                style="position: absolute; top:25px; left:25px; z-index: 1;"></canvas>
        <canvas v-if="channel_view_3" id="layer2" width="600" height="400"
                style="position: absolute; top:25px; left:25px; z-index: 2;"></canvas>
      </div>
    </v-main>
    <v-container
        class="px-0"
        fluid
    >
      <v-switch
          v-model="channel_view_1"
          :label="`Switch 1: ${channel_view_1}`"
      ></v-switch>
      <v-switch
          v-model="channel_view_2"
          :label="`Switch 1: ${channel_view_2}`"
      ></v-switch>
      <v-switch
          v-model="channel_view_3"
          :label="`Switch 1: ${channel_view_3}`"
      ></v-switch>
    </v-container>
    <div style="max-width: 500px">
      <v-slider
          style="margin-left: 15px; z-index: 99"
          width=100
          v-model="fov_num"
          :max="fov_max"
          label="FOV"
          class="align-center"
          v-on:change="drawCanvases"
          color="purple dark-1"
      >
      </v-slider>
      <v-slider
          style="margin-left: 15px; z-index: 99"
          v-model="frame_num"
          :max="frame_max"
          label="FRAME"
          class="align-center"
          v-on:change="drawCanvases"
          color="purple dark-1"
      >
      </v-slider>
    </div>

  </v-app>
</template>


<script>
import Vue from 'vue';
import axios from 'axios';
import HelloWorld from './components/HelloWorld.vue';
import BCanvas from "@/components/ImageHandling/Canvas";

export default Vue.extend({
  name: 'App',

  components: {
    HelloWorld,
    BCanvas,
  },

  data: () => ({
    frame_num: 0,
    frame_max: 0,
    fov_num: 0,
    fov_max: 0,
    connection: null,
    drawer: null,
    vueCanvas: null,
    rectWidth: 200,
    red: 64,
    loaded: false,
    frame_width: null,
    frame_height: null,
    frame_url: null,
    canvas_one_url: null,
    canvas_two_url: null,
    canvas_three_url: null,
    filename: null,
    channel_view_1: true,
    channel_view_2: true,
    channel_view_3: true,
  }),
  methods: {
    forceRerender: function () {

    },
    sendMessage: function (message) {
      console.log(message)
      this.connection.send(message);
    },
    sendFilepath: function (message) {
      let filepath = message.path
      axios.get(`http://localhost:8000/load_dataset/${filepath}`)
          .then((response) => {
            console.log(response.data);
            console.log(response.status);
            console.log(response.statusText);
            console.log(response.headers);
            console.log(response.config);
          });
      let msg = {
        command: "LOAD_FRAMESERIES",
        params: {
          fp: filepath
        },
      };
      this.connection.send(JSON.stringify(msg))
      this.loaded = true;
    },
    fetchFrameSize: function (event) {
      let shape_request = {
        command: "FETCH_FRAME_SHAPE",
        params: {},
      };
      this.connection.send(JSON.stringify(shape_request))
      this.connection.addEventListener('message', function (event) {
        let res = JSON.parse(JSON.parse(event.data)[0]);
        console.log(res)
        this.frame = res.ret
      })
    },
    drawCanvases: function() {
      this.drawFrame('layer0', 0)
      this.drawFrame('layer1', 1)
      this.drawFrame('layer2', 2)
    },
    drawFrame: function (layernum, index) {
      const fetch_url = `http://localhost:8000/fetch_frame/${this.fov_num}/${this.frame_num}/${index}`
      console.log(fetch_url)
      axios.get(fetch_url)
          .then(() => {
            let c = document.getElementById(layernum);
            let ctx = c.getContext("2d");
            let img = new Image;
            img.onload = function () {
              ctx.drawImage(img, 0, 0, 500, 500);
            };
            img.src = fetch_url
          });
    },
    drawRect() {
      // clear canvas
      this.vueCanvas.clearRect(0, 0, 400, 200);

      // draw rect
      this.vueCanvas.beginPath();
      this.vueCanvas.rect(20, 20, this.rectWidth, 100);
      this.vueCanvas.stroke();
    },
    addWidth() {
      this.rectWidth += 10
      this.drawRect()
    },
    subWidth() {
      this.rectWidth -= 10
      this.drawRect()
    }
  },
  created: function () {
    console.log("Starting connection to WebSocket Server")
    this.connection = new WebSocket("ws://localhost:8765")

    this.connection.onmessage = function (event) {
      console.log(event);
    }

    this.connection.onopen = function (event) {
      console.log("Successfully connected to the echo websocket server...")
    }
  },
  mounted() {
    axios.get(`http://localhost:8000/get_fov_size/`)
        .then((response) => {
              let data = response.data
              this.fov_max = data['fov_size']
            }
        )
    axios.get(`http://localhost:8000/get_xy_size/0`)
        .then((response) => {
              let data = response.data
              this.frame_max = data['xy_size']
            }
        )
  },
});
</script>
