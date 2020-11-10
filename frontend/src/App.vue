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
      <BCanvas></BCanvas>
    </v-main>

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

  data: () => ({ drawer: null }),
});
</script>
