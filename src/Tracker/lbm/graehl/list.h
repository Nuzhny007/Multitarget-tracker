// singly linked list, implemented by Jon Graehl <jongraehl@earthling.net>

#include <iostream>

// added this line because without it g++ no longer likes friend declaration
// in struct Node... D. Eppstein, 27 May 1999
template<class T> class ListIter;

template <class T> struct Node {
  T data;
  Node<T> *next;
  Node(const T &it) : data(it), next(NULL) { }
  //  Node(Node<T> *ne) : next(ne) { }
  Node(const T &it, Node<T> *ne) : data(it), next(ne) { }
  static Node<T> *freeList;
  static const int newBlocksize;
#ifdef CUSTOMNEW
  void *operator new(size_t s)
  {
    size_t dummy = s;
    dummy = dummy;
    Node<T> *ret, *max;
    if (freeList) {
      ret = freeList;
      freeList = freeList->next;
      return ret;
    }
    freeList = (Node<T> *)::operator new(newBlocksize * sizeof(Node<T>));
    freeList->next = NULL;
    max = freeList + newBlocksize -1;
    for ( ret = freeList++; freeList < max ; ret = freeList++ )
      freeList->next = ret;
    return freeList--;
  }
  void operator delete(void *p)
  {
    Node<T> *e = (Node<T> *)p;
    e->next = freeList;
    freeList = e;
  }
#endif
};

// due to G++ bug, these static variables cannot be initialized by template
//Node<Arc> *Node<Arc>::freeList = NULL;
//const int Node<Arc>::newBlocksize = 64;

template <class T> class List {
  Node<T> *head;
public:
  Node<T> *first() { return head; }
  List() : head(NULL) { }
  ~List()
  {
    Node<T> *next;
    while (head) {
      next = head->next;
      delete head;
      head = next;
    }
  }
  List( const List<T> & l) {
    //   cout << "List copy " << &l << " to " << this << "\n";
    const Node<T> * h = l.head;
    if ( !h )
      head = NULL;
    else {
      Node<T> * i = head = new Node<T>(h->data);
      for ( h = h->next ; h ; h = h->next) {
	i = i->next = new Node<T>(h->data);
      }
    }
  }
  List(const T &it) { head = new Node<T>(it); }
  int notEmpty() { return (head != 0); }
  int isEmpty() { return (head == 0); }
  int length() { int l = 0 ; for ( Node<T> *p = head ; p ; p = p->next ) ++l; return l; }
  operator Node<T> *() { return head; }
  void push(const T &it) {
    head = new Node<T>(it, head);
  }
  T &top() {
    return head->data;
  }
  void pop() {
    Node<T> *n = head->next;
    delete head;
    head = n;
  }
  friend class ListIter<T>;
};

template<class T> class ListIter {
  Node<T> **previous;
  Node<T> *current;
public:
  ListIter(List<T> &l) : previous(&l.head), current(l.head) {}
  Node<T> *operator ++ () {
    previous = &(current->next);
    current = *previous;
    return current;
  }
  operator int() const { return (current != NULL); }
  T & data() const { return current->data; }
//  T * operator ->() const { return &current->data; }
  Node<T> * cut() {
    Node<T> * ret = current;
    *previous = current = current->next;
    return ret;
  }
  void remove() {
    delete cut();
  }
  T & insert(const T& t) {
    Node<T> *n = new Node<T> (t, current);
    *previous = n;
    previous = &n->next;
    return n->data;
  }
};

template <class T> std::ostream & operator << (std::ostream &out, List<T> &list)
{
  out << "(";
  for ( Node<T> *n = list.first(); n ; n = n->next ) {
    out << n->data;
    if ( n->next )
      out << " ";
  }
  out << ")";
  return out;
}
