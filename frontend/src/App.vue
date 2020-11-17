<template>
  <v-app id="inspire">
    <v-system-bar app>
      <v-spacer></v-spacer>

      <v-icon>mdi-square</v-icon>

      <v-icon>mdi-circle</v-icon>

      <v-icon>mdi-triangle</v-icon>
    </v-system-bar>

    <v-app-bar
        app
        clipped-right
        flat
        height="72"
    >
      <v-responsive max-width="255">
        <v-file-input
            label="File input"
            outlined
            dense
            v-on:change="sendFilepath"
        ></v-file-input>
      </v-responsive>
      <v-spacer></v-spacer>

      <v-responsive max-width="156">
        <v-text-field
            dense
            flat
            hide-details
            rounded
            solo-inverted
        ></v-text-field>
      </v-responsive>
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

    <v-main>
      <button v-on:click="sendMessage('hello')">Send Message</button>
      <canvas id="c"></canvas>
      <div>
        <button @click="drawRect">Add Rect</button>
        <button @click="subWidth">-</button>
        <button @click="addWidth">+</button>
      </div>
      <div>
        <button @click="fetchFrame">Fetch Frame</button>
      </div>
    </v-main>
    <v-slider
        v-model="fov_num"
        :max="2000"
        label="FOV"
        class="align-center"
    >
    </v-slider>
    <v-slider
        v-model="frame_num"
        :max="2000"
        label="FRAME"
        class="align-center"
    >
    </v-slider>


    <v-footer
        app
        color="transparent"
        height="72"
        inset
    >
      <v-text-field
          background-color="grey lighten-1"
          dense
          flat
          hide-details
          rounded
          solo
      ></v-text-field>
    </v-footer>
  </v-app>
</template>


<script>
import Vue from 'vue';
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
    fov_num: 0,
    connection: null,
    drawer: null,
    vueCanvas: null,
    rectWidth: 200,
    red: 64,
    loaded: false,
    frame_width: null,
    frame_height: null,

  }),
  methods: {
    sendMessage: function(message) {
      console.log(message)
      this.connection.send(message);
    },
    sendFilepath: function(message) {
      let filepath = message.path
      // And I image we'll want logging about incoming messages and the like.
      let msg = {
        command: "LOAD_FRAMESERIES",
        params: {
          fp: filepath
        },
      };
      this.connection.send(JSON.stringify(msg))

    },
    fetchFrameSize: function(event) {
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
    }
    fetchFrame: function(event) {
      let array_request = {
        command: "FETCH_FRAME",
        params: {
          fov_num: this.fov_num,
          frame_num: this.frame_num,
        },
      };
      let byte_array = null
      let frame_shape = null
      let frame_dtype = null
      this.connection.send(JSON.stringify(array_request))
      this.connection.addEventListener('message', function (event) {
        let res = JSON.parse(JSON.parse(event.data)[0]);
        console.log(res)
        byte_array = res.ret
      })
      let shape_request = {
        command: "FETCH_FRAME_SHAPE",
        params: {},
      };
      this.connection.send(JSON.stringify(shape_request))
      this.connection.addEventListener('message', function (event) {
        let res = JSON.parse(JSON.parse(event.data)[0]);
        console.log(res)
        frame_shape = res.ret
      })
      let dtype_request = {
        command: "FETCH_FRAME_DTYPE",
        params: {},
      };
      this.connection.send(JSON.stringify(dtype_request))
      this.connection.addEventListener('message', function (event) {
        let res = JSON.parse(JSON.parse(event.data)[0]);
        console.log(res)
        frame_dtype = res.ret
      })
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
  created: function() {
    console.log("Starting connection to WebSocket Server")
    this.connection = new WebSocket("ws://localhost:8765")

    this.connection.onmessage = function(event) {
      console.log(event);
    }

    this.connection.onopen = function(event) {
      console.log(event)
      console.log("Successfully connected to the echo websocket server...")
    }
  },
  mounted() {
    let c = document.getElementById("c");
    let ctx = c.getContext("2d");
    c.width = 1380;
    c.height = 1080;
    this.vueCanvas = ctx;
  },
});
</script>
