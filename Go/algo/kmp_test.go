package algo

import (
	"fmt"
	"testing"
)

func TestKmp(t *testing.T) {
	fmt.Println(kmp("amybmycmy", "my"))
	fmt.Println(kmpSearch("abcda", "a"))
}
